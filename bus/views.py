"""
Views for the Syrian Bus Route Assistant API
"""
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.cache import cache
from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page
from django.views.decorators.vary import vary_on_cookie
from django.core.exceptions import ValidationError
from django.http import Http404
from django.db import connection
from django.core.cache import cache
from django.utils import timezone
from datetime import datetime, timedelta
import json
import logging
from functools import wraps
from .services import BusRouteService
from .validators import validate_coordinates, validate_route_preferences
from .exceptions import (
    InvalidCoordinatesError, 
    NoRouteFoundError, 
    ValidationError as CustomValidationError,
    DatabaseConnectionError,
    ServiceUnavailableError
)
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .utils import clean_mongo_document
from .graph_routing import bfs_multi_leg, nearest_point_on_line, distance, calculate_forward_bus_distance, is_valid_route_segment
from .validators import validate_api_request, validate_route_parameters, ValidationError, handle_validation_error
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

logger = logging.getLogger(__name__)

# Rate limiting configuration
RATE_LIMIT_ENABLED = getattr(settings, 'RATE_LIMIT_ENABLED', True)
RATE_LIMIT_RATE = getattr(settings, 'RATE_LIMIT_RATE', '100/h')  # 100 requests per hour
RATE_LIMIT_BURST = getattr(settings, 'RATE_LIMIT_BURST', 20)  # Allow burst of 20 requests

def rate_limit(key_func=None, rate=RATE_LIMIT_RATE, burst=RATE_LIMIT_BURST):
    """
    Rate limiting decorator for API endpoints.
    
    Args:
        key_func: Function to generate rate limit key (default: IP address)
        rate: Rate limit string (e.g., '100/h', '1000/d')
        burst: Maximum burst requests allowed
    """
    def decorator(view_func):
        @wraps(view_func)
        def wrapped_view(request, *args, **kwargs):
            if not RATE_LIMIT_ENABLED:
                return view_func(request, *args, **kwargs)
            
            # Generate rate limit key
            if key_func:
                key = key_func(request)
            else:
                # Default: use IP address
                x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
                if x_forwarded_for:
                    ip = x_forwarded_for.split(',')[0].strip()
                else:
                    ip = request.META.get('REMOTE_ADDR', 'unknown')
                key = f"rate_limit:{ip}"
            
            # Parse rate limit
            if '/' in rate:
                limit, period = rate.split('/')
                limit = int(limit)
                if period == 's':
                    window = 1
                elif period == 'm':
                    window = 60
                elif period == 'h':
                    window = 3600
                elif period == 'd':
                    window = 86400
                else:
                    window = 3600  # Default to hour
            else:
                limit = int(rate)
                window = 3600  # Default to hour
            
            # Check current usage
            current_time = timezone.now()
            cache_key = f"{key}:{current_time.timestamp() // window}"
            current_count = cache.get(cache_key, 0)
            
            if current_count >= limit:
                logger.warning(f"Rate limit exceeded for {key}: {current_count}/{limit}")
                return JsonResponse({
                    'error': 'Rate limit exceeded',
                    'message': f'Too many requests. Limit: {limit} per {period if period else "hour"}',
                    'retry_after': window
                }, status=429)
            
            # Increment counter
            cache.set(cache_key, current_count + 1, window)
            
            return view_func(request, *args, **kwargs)
        return wrapped_view
    return decorator

def generate_filtered_polyline(coordinates, min_distance_m=400):
    """
    Generate a filtered polyline that excludes segments shorter than min_distance_m.
    
    Args:
        coordinates: List of [lng, lat] coordinate pairs
        min_distance_m: Minimum distance in meters for a segment to be included
    
    Returns:
        List of filtered coordinates
    """
    if len(coordinates) < 2:
        return coordinates
    
    filtered_coords = [coordinates[0]]  # Always include first point
    last_included = coordinates[0]
    
    for i in range(1, len(coordinates)):
        current = coordinates[i]
        
        # Calculate distance from last included point to current point
        distance_m = distance(last_included, current)
        
        # Include point if distance is >= min_distance_m or it's the last point
        if distance_m >= min_distance_m or i == len(coordinates) - 1:
            filtered_coords.append(current)
            last_included = current
    
    return filtered_coords

def generate_route_polylines(legs, origin, dest, route_result=None):
    """
    Generate both full and filtered polylines for a route.
    
    Args:
        legs: List of route legs with type, coordinates, board, alight, bus_distance_m
        origin: Origin coordinates [lng, lat]
        dest: Destination coordinates [lng, lat]
        route_result: Optional route result from bfs_multi_leg containing filtered_polyline
    
    Returns:
        Tuple of (full_polyline, filtered_polyline)
    """
    full_polyline = []
    filtered_polyline = []
    
    # Add origin point to full polyline
    full_polyline.append(origin)
    
    # Process each leg
    for i, leg in enumerate(legs):
        leg_type = leg.get('type', 'bus')
        coords = leg.get('coordinates', [])
        board = leg.get('board')
        alight = leg.get('alight')
        bus_distance_m = leg.get('bus_distance_m', 0)
        
        # Add all coordinates from this leg to full polyline
        if coords:
            full_polyline.extend(coords)
        
        # For filtered polyline, only include bus segments with sufficient distance
        if leg_type == 'bus' and bus_distance_m >= 400 and coords:
            # Find the segment of coordinates between board and alight points
            bus_segment = extract_bus_segment(coords, board, alight)
            if bus_segment:
                filtered_polyline.extend(bus_segment)
    
    # Add destination point to full polyline
    full_polyline.append(dest)
    
    # If route_result contains filtered_polyline, use it instead
    if route_result and 'filtered_polyline' in route_result:
        filtered_polyline = route_result['filtered_polyline']
    elif not filtered_polyline:
        # If no bus segments were added to filtered polyline, use the full polyline
        filtered_polyline = full_polyline.copy()
    
    return full_polyline, filtered_polyline

def extract_bus_segment(coordinates, board_point, alight_point):
    """
    Extract the bus segment between board and alight points.
    
    Args:
        coordinates: List of [lng, lat] coordinate pairs for the bus route
        board_point: Board point [lng, lat]
        alight_point: Alight point [lng, lat]
    
    Returns:
        List of coordinates representing the bus segment
    """
    if not coordinates or not board_point or not alight_point:
        return coordinates
    
    # Find the closest points to board and alight
    board_index = find_closest_point_index(coordinates, board_point)
    alight_index = find_closest_point_index(coordinates, alight_point)
    
    if board_index == -1 or alight_index == -1:
        return coordinates
    
    # Ensure board_index is before alight_index
    if board_index > alight_index:
        board_index, alight_index = alight_index, board_index
    
    # Extract the segment
    return coordinates[board_index:alight_index + 1]

def find_closest_point_index(coordinates, target_point):
    """
    Find the index of the coordinate closest to the target point.
    
    Args:
        coordinates: List of [lng, lat] coordinate pairs
        target_point: Target point [lng, lat]
    
    Returns:
        Index of the closest coordinate, or -1 if not found
    """
    if not coordinates or not target_point:
        return -1
    
    min_distance = float('inf')
    closest_index = -1
    
    for i, coord in enumerate(coordinates):
        if len(coord) >= 2:
            dist = distance(coord, target_point)
            if dist < min_distance:
                min_distance = dist
                closest_index = i
    
    return closest_index

def generate_direct_route_filtered_polyline(coords, entry_point, exit_point):
    """
    Generate filtered polyline for direct routes.
    
    Args:
        coords: List of coordinates for the bus line
        entry_point: Entry point [lng, lat]
        exit_point: Exit point [lng, lat]
    
    Returns:
        List of coordinates representing the bus segment
    """
    if not coords or not entry_point or not exit_point:
        return coords
    
    # Find closest points to entry and exit
    entry_index = find_closest_point_index(coords, entry_point)
    exit_index = find_closest_point_index(coords, exit_point)
    
    if entry_index == -1 or exit_index == -1:
        return coords
    
    # Ensure proper order
    start_idx = min(entry_index, exit_index)
    end_idx = max(entry_index, exit_index)
    
    return coords[start_idx:end_idx + 1]

@api_view(['GET'])
@rate_limit()
def find_route(request):
    """
    Find bus routes between two locations
    
    Query Parameters:
    - from_lng: Starting longitude
    - from_lat: Starting latitude  
    - to_lng: Destination longitude
    - to_lat: Destination latitude
    - preference: Route preference (closest, fewest_walk, fastest)
    - max_routes: Maximum number of routes to return (default: 3)
    - page: Page number for pagination (optional)
    - page_size: Number of routes per page (optional, default: 3)
    - lang: 'ar' or 'en' for route name localization (optional)
    
    Returns:
    - List of route options with details and recommendations
    """
    try:
        # Validate required parameters
        required_params = ['from_lng', 'from_lat', 'to_lng', 'to_lat']
        for param in required_params:
            if param not in request.GET:
                return Response(
                    {"error": f"Missing required parameter: {param}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        # Parse and validate coordinates
        try:
            from_lng = float(request.GET.get("from_lng"))
            from_lat = float(request.GET.get("from_lat"))
            to_lng = float(request.GET.get("to_lng"))
            to_lat = float(request.GET.get("to_lat"))
        except ValueError:
            return Response(
                {"error": "Invalid coordinate values. All coordinates must be valid numbers."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate coordinate ranges
        if not (-180 <= from_lng <= 180 and -90 <= from_lat <= 90):
            return Response(
                {"error": "Invalid starting coordinates. Longitude must be between -180 and 180, latitude between -90 and 90."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if not (-180 <= to_lng <= 180 and -90 <= to_lat <= 90):
            return Response(
                {"error": "Invalid destination coordinates. Longitude must be between -180 and 180, latitude between -90 and 90."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get optional parameters
        preference = request.GET.get("preference", "closest")
        max_routes = int(request.GET.get("max_routes", 3))
        page = int(request.GET.get("page", 1))
        page_size = int(request.GET.get("page_size", max_routes))
        lang = request.GET.get("lang", None)
        
        # Validate preference
        valid_preferences = ["closest", "fewest_walk", "fastest"]
        if preference not in valid_preferences:
            return Response(
                {"error": f"Invalid preference. Must be one of: {', '.join(valid_preferences)}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # SWAP user_location and destination
        user_location = [to_lng, to_lat]
        destination = [from_lng, from_lat]
        
        # Use service layer to find routes
        service = BusRouteService()
        all_routes = service.find_routes(user_location, destination, preference, 1000)  # get all, paginate below
        service.close_connection()
        
        # Pagination
        total_routes = len(all_routes)
        start = (page - 1) * page_size
        end = start + page_size
        paginated_routes = all_routes[start:end]
        # Deduplicate routes by line_id to avoid showing the same route multiple times
        seen_line_ids = set()
        unique_routes = []
        for route in paginated_routes:
            if route.get('line_id') not in seen_line_ids:
                seen_line_ids.add(route.get('line_id'))
                unique_routes.append(route)
        paginated_routes = unique_routes
        # Localization
        if lang in ("ar", "en"):
            for route in paginated_routes:
                if lang == "ar" and route.get("route_name"):
                    route["display_name"] = route["route_name"]
                elif lang == "en" and route.get("route_name_en"):
                    route["display_name"] = route["route_name_en"]
                else:
                    route["display_name"] = route.get("route_name") or route.get("route_name_en")
        # Clean ObjectId for JSON serialization
        paginated_routes = [clean_mongo_document(route) for route in paginated_routes]
        if paginated_routes:
            return Response({
                "success": True,
                "routes": paginated_routes,
                "total_routes_found": len(unique_routes),
                "page": page,
                "page_size": page_size,
                "preference_used": preference
            })
        else:
            # If no direct routes, try two-leg routing
            service = BusRouteService()
            # Find all routes near origin
            origin_routes = service.find_routes(user_location, user_location, preference="closest", max_routes=100)
            # Find all routes near destination
            dest_routes = service.find_routes(destination, destination, preference="closest", max_routes=100)
            service.close_connection()
            # Try to find transfer points between any pair
            multi_leg_suggestions = []
            for r1 in origin_routes:
                for r2 in dest_routes:
                    # Find closest points between the two routes
                    coords1 = r1["coordinates"]
                    coords2 = r2["coordinates"]
                    for p1 in coords1:
                        for p2 in coords2:
                            transfer_dist = service.distance_between_points(p1, p2)
                            if transfer_dist < 200:  # 200 meters threshold for transfer
                                # Total walking: user to r1 entry + transfer + r2 exit to dest
                                total_walk = r1["distance_to_entry"] + transfer_dist + r2["distance_to_exit"]
                                multi_leg_suggestions.append({
                                    "legs": [
                                        {
                                            "from": user_location,
                                            "to": p1,
                                            "route": clean_mongo_document(r1)
                                        },
                                        {
                                            "from": p2,
                                            "to": destination,
                                            "route": clean_mongo_document(r2)
                                        }
                                    ],
                                    "transfer_point": p1,
                                    "transfer_distance": transfer_dist,
                                    "total_walking_distance": total_walk
                                })
            # Sort by total walking distance and take top 3
            multi_leg_suggestions.sort(key=lambda x: x["total_walking_distance"])
            multi_leg_suggestions = multi_leg_suggestions[:3]
            # Deduplicate multi-leg suggestions to avoid showing the same route combinations
            seen_combinations = set()
            unique_suggestions = []
            for suggestion in multi_leg_suggestions:
                # Create a unique key for this combination of routes
                route_ids = tuple(sorted([
                    suggestion["legs"][0]["route"]["line_id"],
                    suggestion["legs"][1]["route"]["line_id"]
                ]))
                if route_ids not in seen_combinations:
                    seen_combinations.add(route_ids)
                    unique_suggestions.append(suggestion)
            multi_leg_suggestions = unique_suggestions
            if multi_leg_suggestions:
                return Response({
                    "success": True,
                    "multi_leg": True,
                    "suggestions": multi_leg_suggestions,
                    "total_suggestions": len(multi_leg_suggestions),
                    "message": "No direct route found, but here are multi-bus options:"
                })
            return Response({
                "success": False,
                "message": "No suitable routes found. Try adjusting your location or destination.",
                "routes": [],
                "total_routes_found": total_routes,
                "page": page,
                "page_size": page_size
            }, status=status.HTTP_404_NOT_FOUND)
            
    except Exception as e:
        logger.error(f"Error in find_route: {e}")
        return Response(
            {"error": "Internal server error. Please try again later."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@rate_limit()
def route_statistics(request):
    """
    Get statistics about available routes in the database
    
    Returns:
    - Basic statistics about the route database
    """
    try:
        service = BusRouteService()
        stats = service.get_route_statistics()
        service.close_connection()
        
        return Response({
            "success": True,
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error in route_statistics: {e}")
        return Response(
            {"error": "Internal server error. Please try again later."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@rate_limit()
def health_check(request):
    """
    Health check endpoint to verify API is running
    
    Returns:
    - API status and basic information
    """
    try:
        from .mongo import initialize_mongo
        from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
        
        # Initialize MongoDB connection
        client, db, collection = initialize_mongo()
        
        # Test MongoDB connection by listing collections
        collections = db.list_collection_names()
        
        # Get basic stats
        total_routes = collection.count_documents({})
        
        return Response({
            "status": "healthy",
            "service": "Syrian Bus Route Assistant API",
            "version": "1.0.0",
            "database_connected": True,
            "mongodb_collections": collections,
            "total_routes": total_routes
        })
        
    except ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB connection timeout in health check: {e}")
        return Response({
            "status": "unhealthy",
            "error": "MongoDB connection timeout"
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failure in health check: {e}")
        return Response({
            "status": "unhealthy",
            "error": "MongoDB connection failed"
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        
    except Exception as e:
        logger.error(f"Error in health_check: {e}")
        return Response({
            "status": "unhealthy",
            "error": f"Database connection failed: {str(e)}"
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)

@api_view(['GET'])
@rate_limit()
def diagnostics(request):
    """
    Diagnostics endpoint for debugging and health checks.
    Returns environment, settings, and sample route data.
    """
    try:
        from .mongo import initialize_mongo
        from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
        
        # Initialize MongoDB connection
        client, db, collection = initialize_mongo()
        
        # Test MongoDB connection by listing collections
        collections = db.list_collection_names()
        
        # Get basic stats
        total_routes = collection.count_documents({})
        
        # Get a sample route
        sample_route = collection.find_one({})
        
        return Response({
            "status": "ok",
            "settings": {
                "mongodb_uri": getattr(settings, "MONGO_URI", None),
                "mongodb_database": getattr(settings, "MONGODB_DATABASE", None),
                "mongodb_collection": getattr(settings, "MONGODB_COLLECTION", None),
            },
            "statistics": {
                "total_routes": total_routes,
                "collections": collections
            },
            "sample_route": clean_mongo_document(sample_route) if sample_route else None
        })
        
    except ServerSelectionTimeoutError as e:
        logger.error(f"MongoDB connection timeout in diagnostics: {e}")
        return Response({
            "status": "error",
            "error": "MongoDB connection timeout"
        }, status=500)
        
    except ConnectionFailure as e:
        logger.error(f"MongoDB connection failure in diagnostics: {e}")
        return Response({
            "status": "error",
            "error": "MongoDB connection failed"
        }, status=500)
        
    except Exception as e:
        logger.error(f"Error in diagnostics: {e}")
        return Response({"status": "error", "error": str(e)}, status=500)

@api_view(['GET'])
@rate_limit()
def top_suggestions(request):
    """
    Returns the top route for each category: closest, fastest, and fewest_walk.
    Accepts the same parameters as find_route.
    """
    try:
        required_params = ['from_lng', 'from_lat', 'to_lng', 'to_lat']
        for param in required_params:
            if param not in request.GET:
                return Response(
                    {"error": f"Missing required parameter: {param}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        try:
            from_lng = float(request.GET.get("from_lng"))
            from_lat = float(request.GET.get("from_lat"))
            to_lng = float(request.GET.get("to_lng"))
            to_lat = float(request.GET.get("to_lat"))
        except ValueError:
            return Response(
                {"error": "Invalid coordinate values. All coordinates must be valid numbers."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not (-180 <= from_lng <= 180 and -90 <= from_lat <= 90):
            return Response(
                {"error": "Invalid starting coordinates. Longitude must be between -180 and 180, latitude between -90 and 90."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not (-180 <= to_lng <= 180 and -90 <= to_lat <= 90):
            return Response(
                {"error": "Invalid destination coordinates. Longitude must be between -180 and 180, latitude between -90 and 90."},
                status=status.HTTP_400_BAD_REQUEST
            )
        lang = request.GET.get("lang", None)
        user_location = [from_lng, from_lat]
        destination = [to_lng, to_lat]
        service = BusRouteService()
        all_routes = service.find_routes(user_location, destination, preference="closest", max_routes=1000)
        service.close_connection()
        # Prepare top suggestions for each category
        suggestions = {}
        categories = ["closest", "fewest_walk"]
        for category in categories:
            if category == "closest":
                # Use total_walking_distance for closest (minimize total walking)
                sorted_routes = sorted(all_routes, key=lambda r: r["total_walking_distance"])
            elif category == "fewest_walk":
                sorted_routes = sorted(all_routes, key=lambda r: r["total_walking_distance"])
            else:
                sorted_routes = []
            # Pick the first unique route for this category
            for route in sorted_routes:
                # Optionally localize display_name
                if lang in ("ar", "en"):
                    if lang == "ar" and route.get("route_name"):
                        route["display_name"] = route["route_name"]
                    elif lang == "en" and route.get("route_name_en"):
                        route["display_name"] = route["route_name_en"]
                    else:
                        route["display_name"] = route.get("route_name") or route.get("route_name_en")
                from .utils import clean_mongo_document
                suggestions[category] = clean_mongo_document(route)
                break  # Only the top one
            else:
                suggestions[category] = None
        return Response({
            "success": True,
            "suggestions": suggestions
        })
    except Exception as e:
        logger.error(f"Error in top_suggestions: {e}")
        return Response(
            {"error": "Internal server error. Please try again later."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@rate_limit()
def suggest_route(request):
    """
    Returns the top route for the user-specified category (closest or fewest_walk).
    Query params: from_lng, from_lat, to_lng, to_lat, category (closest|fewest_walk), lang (optional)
    """
    try:
        required_params = ['from_lng', 'from_lat', 'to_lng', 'to_lat', 'category']
        for param in required_params:
            if param not in request.GET:
                return Response(
                    {"error": f"Missing required parameter: {param}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        try:
            from_lng = float(request.GET.get("from_lng"))
            from_lat = float(request.GET.get("from_lat"))
            to_lng = float(request.GET.get("to_lng"))
            to_lat = float(request.GET.get("to_lat"))
        except ValueError:
            return Response(
                {"error": "Invalid coordinate values. All coordinates must be valid numbers."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not (-180 <= from_lng <= 180 and -90 <= from_lat <= 90):
            return Response(
                {"error": "Invalid starting coordinates. Longitude must be between -180 and 180, latitude between -90 and 90."},
                status=status.HTTP_400_BAD_REQUEST
            )
        if not (-180 <= to_lng <= 180 and -90 <= to_lat <= 90):
            return Response(
                {"error": "Invalid destination coordinates. Longitude must be between -180 and 180, latitude between -90 and 90."},
                status=status.HTTP_400_BAD_REQUEST
            )
        category = request.GET.get("category")
        lang = request.GET.get("lang", None)
        valid_categories = ["closest", "fewest_walk"]
        if category not in valid_categories:
            return Response(
                {"error": f"Invalid category. Must be one of: {', '.join(valid_categories)}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        # SWAP user_location and destination
        user_location = [to_lng, to_lat]
        destination = [from_lng, from_lat]
        service = BusRouteService()
        all_routes = service.find_routes(user_location, destination, preference="closest", max_routes=1000)
        service.close_connection()
        # Sort and select top route for the chosen category
        if category == "closest":
            sorted_routes = sorted(all_routes, key=lambda r: r["total_walking_distance"])
        elif category == "fewest_walk":
            sorted_routes = sorted(all_routes, key=lambda r: r["total_walking_distance"])
        else:
            sorted_routes = []
        top_route = None
        for route in sorted_routes:
            if lang in ("ar", "en"):
                if lang == "ar" and route.get("route_name"):
                    route["display_name"] = route["route_name"]
                elif lang == "en" and route.get("route_name_en"):
                    route["display_name"] = route["route_name_en"]
                else:
                    route["display_name"] = route.get("route_name") or route.get("route_name_en")
            from .utils import clean_mongo_document
            top_route = clean_mongo_document(route)
            break
        # If no direct route, try two-leg routing
        if not top_route:
            # Find all routes near origin
            origin_routes = service.find_routes(user_location, user_location, preference="closest", max_routes=100)
            # Find all routes near destination
            dest_routes = service.find_routes(destination, destination, preference="closest", max_routes=100)
            # Try to find a transfer point between any pair
            min_total_walk = float('inf')
            best_combo = None
            for r1 in origin_routes:
                for r2 in dest_routes:
                    # Find closest points between the two routes
                    coords1 = r1["coordinates"]
                    coords2 = r2["coordinates"]
                    for p1 in coords1:
                        for p2 in coords2:
                            transfer_dist = service.distance_between_points(p1, p2)
                            if transfer_dist < 200:  # 200 meters threshold for transfer
                                # Total walking: user to r1 entry + transfer + r2 exit to dest
                                total_walk = r1["distance_to_entry"] + transfer_dist + r2["distance_to_exit"]
                                if total_walk < min_total_walk:
                                    min_total_walk = total_walk
                                    best_combo = (r1, p1, r2, p2, transfer_dist)
            if best_combo:
                r1, p1, r2, p2, transfer_dist = best_combo
                from .utils import clean_mongo_document
                return Response({
                    "success": True,
                    "multi_leg": True,
                    "legs": [
                        {
                            "from": user_location,
                            "to": p1,
                            "route": clean_mongo_document(r1)
                        },
                        {
                            "from": p2,
                            "to": destination,
                            "route": clean_mongo_document(r2)
                        }
                    ],
                    "transfer_point": p1,
                    "transfer_distance": transfer_dist,
                    "total_walking_distance": min_total_walk
                })
        return Response({
            "success": True,
            "category": category,
            "route": top_route
        })
    except Exception as e:
        logger.error(f"Error in suggest_route: {e}")
        return Response(
            {"error": "Internal server error. Please try again later."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['GET'])
@rate_limit()
@validate_api_request(validate_route_parameters)
@swagger_auto_schema(
    operation_description="Find optimal bus routes between two points in Syria",
    operation_summary="Find bus routes",
    manual_parameters=[
        openapi.Parameter(
            'from_lng',
            openapi.IN_QUERY,
            description="Origin longitude (required)",
            type=openapi.TYPE_NUMBER,
            required=True,
            example=36.297949
        ),
        openapi.Parameter(
            'from_lat',
            openapi.IN_QUERY,
            description="Origin latitude (required)",
            type=openapi.TYPE_NUMBER,
            required=True,
            example=33.537161
        ),
        openapi.Parameter(
            'to_lng',
            openapi.IN_QUERY,
            description="Destination longitude (required)",
            type=openapi.TYPE_NUMBER,
            required=True,
            example=36.288504
        ),
        openapi.Parameter(
            'to_lat',
            openapi.IN_QUERY,
            description="Destination latitude (required)",
            type=openapi.TYPE_NUMBER,
            required=True,
            example=33.521974
        ),
        openapi.Parameter(
            'category',
            openapi.IN_QUERY,
            description="Route preference category",
            type=openapi.TYPE_STRING,
            required=False,
            enum=['fewest_walking', 'least_transfers', 'fastest'],
            example='fastest'
        ),
    ],
    responses={
        200: openapi.Response(
            description="Routes found successfully",
            examples={
                "application/json": {
                    "fewest_walking": {
                        "route": ["route_1"],
                        "legs": [
                            {
                                "line_id": "route_1",
                                "name": "Route 1",
                                "type": "bus",
                                "direction": "outbound",
                                "coordinates": [[36.297949, 33.537161], [36.288504, 33.521974]],
                                "board": [36.297949, 33.537161],
                                "alight": [36.288504, 33.521974],
                                "walk_start": 0.0,
                                "walk_end": 0.0,
                                "bus_distance_m": 1500.0,
                                "walk_time_min": 0.0,
                                "bus_time_min": 4.5,
                                "leg_time_min": 4.5
                            }
                        ],
                        "total_walking": 0.0,
                        "total_time_min": 4.5,
                        "origin_to_first_bus_walk": 0.0,
                        "last_bus_to_destination_walk": 0.0,
                        "final_leg_suggestion": "walk",
                        "num_transfers": 0
                    },
                    "least_transfers": {
                        "route": ["route_1"],
                        "legs": [
                            {
                                "line_id": "route_1",
                                "name": "Route 1",
                                "type": "bus",
                                "direction": "outbound",
                                "coordinates": [[36.297949, 33.537161], [36.288504, 33.521974]],
                                "board": [36.297949, 33.537161],
                                "alight": [36.288504, 33.521974],
                                "walk_start": 0.0,
                                "walk_end": 0.0,
                                "bus_distance_m": 1500.0,
                                "walk_time_min": 0.0,
                                "bus_time_min": 4.5,
                                "leg_time_min": 4.5
                            }
                        ],
                        "total_walking": 0.0,
                        "total_time_min": 4.5,
                        "origin_to_first_bus_walk": 0.0,
                        "last_bus_to_destination_walk": 0.0,
                        "final_leg_suggestion": "walk",
                        "num_transfers": 0
                    },
                    "fastest": {
                        "route": ["route_1"],
                        "legs": [
                            {
                                "line_id": "route_1",
                                "name": "Route 1",
                                "type": "bus",
                                "direction": "outbound",
                                "coordinates": [[36.297949, 33.537161], [36.288504, 33.521974]],
                                "board": [36.297949, 33.537161],
                                "alight": [36.288504, 33.521974],
                                "walk_start": 0.0,
                                "walk_end": 0.0,
                                "bus_distance_m": 1500.0,
                                "walk_time_min": 0.0,
                                "bus_time_min": 4.5,
                                "leg_time_min": 4.5
                            }
                        ],
                        "total_walking": 0.0,
                        "total_time_min": 4.5,
                        "origin_to_first_bus_walk": 0.0,
                        "last_bus_to_destination_walk": 0.0,
                        "final_leg_suggestion": "walk",
                        "num_transfers": 0
                    },
                    "message": "Best routes found."
                }
            }
        ),
        400: openapi.Response(
            description="Invalid parameters",
            examples={
                "application/json": {
                    "error": True,
                    "message": "Invalid from_lng: must be a valid number",
                    "code": "INVALID_COORDINATE_TYPE",
                    "field": "from_lng"
                }
            }
        ),
        404: openapi.Response(
            description="No routes found",
            examples={
                "application/json": {
                    "message": "No available bus route to destination. Please use taxi."
                }
            }
        ),
        500: openapi.Response(
            description="Internal server error",
            examples={
                "application/json": {
                    "error": True,
                    "message": "Internal server error. Please try again later.",
                    "code": "INTERNAL_ERROR"
                }
            }
        )
    }
)
def graph_route(request):
    """
    Advanced graph-based multi-leg route search for public transport.
    Returns all possible paths, each with full route details and estimated travel time.
    
    Parameters:
    - from_lng, from_lat: Origin coordinates (required)
    - to_lng, to_lat: Destination coordinates (required)
    - category: Route preference ('fewest_walking', 'closest', or None for all routes)
    
    Returns:
    - JSON response with route options and details
    """
    try:
        # Use validated parameters from decorator
        validated_params = request.validated_params
        origin = [validated_params['from_lng'], validated_params['from_lat']]
        dest = [validated_params['to_lng'], validated_params['to_lat']]
        category = validated_params.get('category')
        
        service = BusRouteService()
        # Use optimized query with projection for better performance
        lines = service.get_routes_for_routing(origin, dest, max_distance_km=50)
        line_map = {line['line_id']: line for line in lines}
        
        # Get min_bus_distance from settings
        bus_settings = getattr(settings, 'BUS_ROUTE_SETTINGS', {})
        min_bus_distance = bus_settings.get('MIN_BUS_DISTANCE_METERS', 400)
        
        # 1. Direct route
        logger.info(f"Checking {len(lines)} lines for direct routes...")
        for line in lines:
            coords = line['route']['coordinates']
            entry, d_entry, entry_index = nearest_point_on_line(origin, coords)
            exit, d_exit, exit_index = nearest_point_on_line(dest, coords)
            logger.debug(f"Direct route check: {line['line_id']} entry_dist={d_entry:.2f}m exit_dist={d_exit:.2f}m")
            if d_entry < 300 and d_exit < 300:
                if is_valid_route_segment(coords, entry_index, exit_index):
                    bus_distance = calculate_forward_bus_distance(coords, entry_index, exit_index)
                    logger.debug(f"Direct route check: {line['line_id']} bus_distance={bus_distance:.2f}m (min required: {min_bus_distance}m)")
                    if bus_distance >= min_bus_distance:
                        # Generate polylines for the route
                        legs = [{
                            "line_id": line['line_id'],
                            "name": line.get('name'),
                            "type": line.get('type'),
                            "direction": line.get('direction'),
                            "coordinates": coords,
                            "board": entry,
                            "alight": exit,
                            "entry_point": entry,
                            "exit_point": exit,
                            "walk_start": d_entry,
                            "walk_end": d_exit,
                            "bus_distance_m": bus_distance,
                            "walk_time_min": (d_entry + d_exit) / 5000 * 60,
                            "bus_time_min": bus_distance / 20000 * 60,
                            "leg_time_min": ((d_entry + d_exit) / 5000 * 60) + (bus_distance / 20000 * 60)
                        }]
                        
                        full_polyline, filtered_polyline = generate_route_polylines(legs, origin, dest)
                        
                        # Create enhanced direct route with Västtrafik features
                        enhanced_direct_route = {
                            "route": [line['line_id']],
                            "legs": legs,
                            "full_polyline": full_polyline,
                            "filtered_polyline": filtered_polyline,
                            "total_walking": d_entry + d_exit,
                            "total_time_min": ((d_entry + d_exit) / 5000 * 60) + (bus_distance / 20000 * 60),
                            "walking_directions": {},
                            "accessibility": {
                                "wheelchair_accessible": True,
                                "elevator_available": False,
                                "stairs_count": 0,
                                "difficulty": "easy"
                            },
                            "real_time_info": {
                                line['line_id']: service.get_real_time_info(line['line_id'])
                            },
                            "frequency_info": {
                                line['line_id']: service.get_line_frequency(line['line_id'])
                            }
                        }
                        logger.info(f"Direct route accepted: {line['line_id']} bus_distance={bus_distance:.2f}m")
                        return Response({
                            "fewest_walking": {
                                "best": enhanced_direct_route,
                                "alternatives": []
                            },
                            "least_transfers": {
                                "best": enhanced_direct_route,
                                "alternatives": []
                            },
                            "fastest": {
                                "best": enhanced_direct_route,
                                "alternatives": []
                            },
                            "message": "Direct route found with Västtrafik enhancements.",
                            "total_routes_found": 1
                        })
                    else:
                        logger.info(f"Direct route rejected: {line['line_id']} bus_distance={bus_distance:.2f}m < {min_bus_distance}m")
                        continue
                else:
                    logger.info(f"Direct route rejected: {line['line_id']} invalid route segment entry_index={entry_index} exit_index={exit_index}")
                    continue
        # 2. Multi-leg route (graph search) - prioritize simpler routes
        # First try with max_legs=2 (1 transfer max)
        results = bfs_multi_leg(lines, origin, dest, entry_thresh=300, exit_thresh=300, transfer_thresh=400, max_legs=2, min_bus_distance=min_bus_distance)
        
        # If no simple routes found, try with max_legs=3 (2 transfers max)
        if not results:
            logger.info("No simple routes found (max 1 transfer), trying with max 2 transfers...")
            results = bfs_multi_leg(lines, origin, dest, entry_thresh=300, exit_thresh=300, transfer_thresh=400, max_legs=3, min_bus_distance=min_bus_distance)
        
        # Only if still no routes, try with max_legs=4 (3 transfers max) - but this should be rare
        if not results:
            logger.info("No routes found with max 2 transfers, trying with max 3 transfers...")
            results = bfs_multi_leg(lines, origin, dest, entry_thresh=300, exit_thresh=300, transfer_thresh=400, max_legs=4, min_bus_distance=min_bus_distance)
        # Filter out routes with legs that have bus distances less than min_bus_distance
        filtered_results = []
        for res in results:
            has_short_leg = False
            for i, line_id in enumerate(res['lines']):
                line = line_map[line_id]
                coords = line['route']['coordinates']
                board = res['entry_points'][i]
                if i + 1 < len(res['entry_points']):
                    alight = res['entry_points'][i+1]
                else:
                    alight = res['exit_point']
                # Find indices for board and alight points with tolerance
                board_index = -1
                alight_index = -1
                tolerance = 0.0001  # ~10 meters tolerance for coordinate matching
                
                for j, coord in enumerate(coords):
                    # Check board point with tolerance
                    if board_index == -1 and abs(coord[0] - board[0]) < tolerance and abs(coord[1] - board[1]) < tolerance:
                        board_index = j
                    # Check alight point with tolerance
                    if alight_index == -1 and abs(coord[0] - alight[0]) < tolerance and abs(coord[1] - alight[1]) < tolerance:
                        alight_index = j
                
                # If still not found, try nearest point approach
                if board_index == -1:
                    board_index = nearest_point_on_line(board, coords)[2]
                if alight_index == -1:
                    alight_index = nearest_point_on_line(alight, coords)[2]
                
                # Skip legs with invalid indices
                if board_index == alight_index or board_index == -1 or alight_index == -1:
                    logger.warning(f"Skipping leg with invalid indices: {line_id} board_index={board_index} alight_index={alight_index}")
                    logger.debug(f"Board point: {board}, Alight point: {alight}")
                    has_short_leg = True
                    break
                # Calculate bus distance
                if board_index != -1 and alight_index != -1 and is_valid_route_segment(coords, board_index, alight_index):
                    bus_distance = calculate_forward_bus_distance(coords, board_index, alight_index)
                    if bus_distance < min_bus_distance:
                        logger.info(f"Filtering out route {res['lines']} due to short leg {line_id}: {bus_distance:.2f}m < {min_bus_distance}m")
                        has_short_leg = True
                        break
                else:
                    has_short_leg = True
                    logger.warning(f"Filtering out route {res['lines']} due to invalid/zero leg: {line_id}")
                    break
            if not has_short_leg:
                filtered_results.append(res)
        logger.info(f"Filtered {len(results) - len(filtered_results)} routes with short bus distances")
        all_paths = []
        seen = set()
        for res in filtered_results:
            key = (
                tuple(res['lines']),
                tuple(tuple(pt) for pt in res['entry_points']),
                tuple(res['exit_point']) if isinstance(res['exit_point'], (list, tuple)) else res['exit_point']
            )
            if key in seen:
                continue
            seen.add(key)
            legs = []
            total_time = 0.0
            for i, line_id in enumerate(res['lines']):
                line = line_map[line_id]
                coords = line['route']['coordinates']
                board = res['entry_points'][i]
                if i + 1 < len(res['entry_points']):
                    alight = res['entry_points'][i+1]
                else:
                    alight = res['exit_point']
                # Find indices for board and alight points with tolerance
                board_index = -1
                alight_index = -1
                tolerance = 0.0001  # ~10 meters tolerance for coordinate matching
                
                for j, coord in enumerate(coords):
                    # Check board point with tolerance
                    if board_index == -1 and abs(coord[0] - board[0]) < tolerance and abs(coord[1] - board[1]) < tolerance:
                        board_index = j
                    # Check alight point with tolerance
                    if alight_index == -1 and abs(coord[0] - alight[0]) < tolerance and abs(coord[1] - alight[1]) < tolerance:
                        alight_index = j
                
                # If still not found, try nearest point approach
                if board_index == -1:
                    board_index = nearest_point_on_line(board, coords)[2]
                if alight_index == -1:
                    alight_index = nearest_point_on_line(alight, coords)[2]
                
                # Skip legs with invalid indices
                if board_index == alight_index or board_index == -1 or alight_index == -1:
                    logger.warning(f"Skipping leg with invalid indices (final build): {line_id} board_index={board_index} alight_index={alight_index}")
                    logger.debug(f"Board point: {board}, Alight point: {alight}")
                    continue
                # Calculate bus distance using forward travel only
                if board_index != -1 and alight_index != -1 and is_valid_route_segment(coords, board_index, alight_index):
                    bus_distance = calculate_forward_bus_distance(coords, board_index, alight_index)
                else:
                    bus_distance = 0.0  # Invalid route segment
                
                # Calculate walking distances correctly
                if i == 0:
                    # First leg: walk from origin to board, and from alight to transfer point
                    walk_start = distance(board, origin)
                    walk_end = distance(alight, res['entry_points'][i+1]) if i + 1 < len(res['entry_points']) else distance(alight, dest)
                else:
                    # Other legs: walk from transfer point to board, and from alight to next transfer or destination
                    # The board point should be the same as the previous alight point (same transfer point)
                    # So walk_start should be 0 since you're already at the transfer point
                    walk_start = 0
                    walk_end = distance(alight, res['entry_points'][i+1]) if i + 1 < len(res['entry_points']) else distance(alight, dest)
                walk_time = (walk_start + walk_end) / 5000 * 60
                bus_time = bus_distance / 20000 * 60
                leg_time = walk_time + bus_time
                total_time += leg_time
                legs.append({
                    "line_id": line_id,
                    "name": line.get('name'),
                    "type": line.get('type'),
                    "direction": line.get('direction'),
                    "coordinates": coords,
                    "board": board,
                    "alight": alight,
                    "entry_point": board,
                    "exit_point": alight,
                    "walk_start": walk_start,
                    "walk_end": walk_end,
                    "bus_distance_m": bus_distance,
                    "walk_time_min": walk_time,
                    "bus_time_min": bus_time,
                    "leg_time_min": leg_time
                })
            # Calculate total walking distance from the actual legs
            total_walk = sum(leg["walk_start"] + leg["walk_end"] for leg in legs)
            # Suggest walk or taxi if final walk is more than the configured threshold
            final_walk = legs[-1]["walk_end"] if legs else 0
            taxi_threshold = getattr(settings, 'TAXI_SUGGESTION_DISTANCE', 1000)
            suggestion = "walk" if final_walk <= taxi_threshold else "taxi recommended"
            
            # Generate Västtrafik-style walking directions
            walking_directions = {}
            if legs:
                # Origin to first bus
                if legs[0]["walk_start"] > 0:
                    origin_to_bus = service.generate_walking_directions(
                        origin, legs[0]["board"]
                    )
                    walking_directions["origin_to_first_bus"] = origin_to_bus
                
                # Last bus to destination
                if final_walk > 0:
                    last_bus_to_dest = service.generate_walking_directions(
                        legs[-1]["alight"], dest
                    )
                    walking_directions["last_bus_to_destination"] = last_bus_to_dest
                
                # Transfer walking directions
                transfer_walks = []
                for i in range(len(legs) - 1):
                    if legs[i]["walk_end"] > 0:
                        transfer_walk = service.generate_walking_directions(
                            legs[i]["alight"], legs[i+1]["board"]
                        )
                        transfer_walks.append({
                            "from_leg": i,
                            "to_leg": i + 1,
                            "directions": transfer_walk
                        })
                if transfer_walks:
                    walking_directions["transfers"] = transfer_walks
            # Generate polylines for the route
            full_polyline, filtered_polyline = generate_route_polylines(legs, origin, dest, res)
            
            all_paths.append({
                "route": res['lines'],
                "legs": legs,
                "full_polyline": full_polyline,
                "filtered_polyline": filtered_polyline,
                "transfers": res['transfers'],
                "entry_points": res['entry_points'],
                "exit_point": res['exit_point'],
                "entry_point": legs[0]["entry_point"] if legs else None,
                "final_exit_point": legs[-1]["exit_point"] if legs else None,
                "exit_walk": res['exit_walk'],
                "total_walking": total_walk,
                "total_time_min": total_time,
                "origin_to_first_bus_walk": legs[0]["walk_start"] if legs else 0,
                "last_bus_to_destination_walk": final_walk,
                "final_leg_suggestion": suggestion,
                "num_transfers": len(res['lines']) - 1,
                "walking_directions": walking_directions,  # Västtrafik-style walking directions
                "accessibility": {
                    "wheelchair_accessible": True,  # Default - can be enhanced with real data
                    "elevator_available": False,
                    "stairs_count": 0,
                    "difficulty": "easy" if total_walk <= 500 else "moderate" if total_walk <= 1000 else "difficult"
                },
                "real_time_info": {
                    line_id: service.get_real_time_info(line_id) for line_id in res['lines']
                },
                "frequency_info": {
                    line_id: service.get_line_frequency(line_id) for line_id in res['lines']
                }
            })
        
        # Sort by different criteria for different categories - Västtrafik style
        # Prioritize simpler routes (fewer transfers) in all categories
        fewest_walking_candidates = sorted(all_paths, key=lambda x: (x["num_transfers"], x["total_walking"]))
        least_transfers_candidates = sorted(all_paths, key=lambda x: (x["num_transfers"], x["total_walking"]))
        fastest_candidates = sorted(all_paths, key=lambda x: (x["num_transfers"], x["total_time_min"]))

        if all_paths:
            # Return only the best route for each category (simpler for passengers)
            response_data = {
                "fewest_walking": {
                    "best": fewest_walking_candidates[0] if fewest_walking_candidates else None,
                    "alternatives": []  # No alternatives to keep it simple
                },
                "least_transfers": {
                    "best": least_transfers_candidates[0] if least_transfers_candidates else None,
                    "alternatives": []  # No alternatives to keep it simple
                },
                "fastest": {
                    "best": fastest_candidates[0] if fastest_candidates else None,
                    "alternatives": []  # No alternatives to keep it simple
                },
                "message": "Best routes found with alternatives.",
                "total_routes_found": len(all_paths)
            }
            return Response(response_data)
        
        # 3. Closest + walk
        for line in lines:
            coords = line['route']['coordinates']
            entry, d_entry, entry_index = nearest_point_on_line(origin, coords)
            exit, d_exit, exit_index = nearest_point_on_line(dest, coords)
            if d_entry < 300 and d_exit < 1000:
                # Check if the route segment is valid (entry before exit)
                if is_valid_route_segment(coords, entry_index, exit_index):
                    bus_distance = calculate_forward_bus_distance(coords, entry_index, exit_index)
                    walk_time = (d_entry + d_exit) / 5000 * 60
                    bus_time = bus_distance / 20000 * 60
                    total_time = walk_time + bus_time
                    legs = [{
                        "line_id": line['line_id'],
                        "name": line.get('name'),
                        "type": line.get('type'),
                        "direction": line.get('direction'),
                        "coordinates": coords,
                        "board": entry,
                        "alight": exit,
                        "walk_start": d_entry,
                        "walk_end": d_exit,
                        "bus_distance_m": bus_distance,
                        "walk_time_min": walk_time,
                        "bus_time_min": bus_time,
                        "leg_time_min": total_time
                    }]
                    
                    # Generate polylines for the route
                    full_polyline, _ = generate_route_polylines(legs, origin, dest)
                    
                    # Generate filtered polyline for direct route
                    filtered_polyline = generate_direct_route_filtered_polyline(
                        coords, entry, exit
                    )
                    
                    # Return enhanced Västtrafik format even for direct routes
                    enhanced_direct_route = {
                        "route": [line['line_id']],
                        "legs": legs,
                        "full_polyline": full_polyline,
                        "filtered_polyline": filtered_polyline,
                        "total_walking": d_entry + d_exit,  
                        "total_time_min": total_time,
                        "walking_directions": {},
                        "accessibility": {
                            "wheelchair_accessible": True,
                            "elevator_available": False,
                            "stairs_count": 0,
                            "difficulty": "easy"
                        },
                        "real_time_info": {
                            line['line_id']: service.get_real_time_info(line['line_id'])
                        },
                        "frequency_info": {
                            line['line_id']: service.get_line_frequency(line['line_id'])
                        }
                    }
                    
                    return Response({
                        "fewest_walking": {
                            "best": enhanced_direct_route,
                            "alternatives": []
                        },
                        "least_transfers": {
                            "best": enhanced_direct_route,
                            "alternatives": []
                        },
                        "fastest": {
                            "best": enhanced_direct_route,
                            "alternatives": []
                        },
                        "message": "Direct route found with Västtrafik enhancements.",
                        "total_routes_found": 1
                    })
        
        # 4. No route
        return Response({
            "message": "No available bus route to destination. Please use taxi."
        }, status=404)
        
    except ValidationError as e:
        return handle_validation_error(e)
    except Exception as e:
        logger.error(f"Error in graph_route: {e}")
        return Response({
            "error": True,
            "message": "Internal server error. Please try again later.",
            "code": "INTERNAL_ERROR"
        }, status=500)

@api_view(['GET'])
@rate_limit()
@swagger_auto_schema(
    operation_description="Get Västtrafik-style real-time information for bus lines",
    operation_summary="Get real-time info",
    manual_parameters=[
        openapi.Parameter(
            'line_id',
            openapi.IN_QUERY,
            description="Bus line ID (required)",
            type=openapi.TYPE_STRING,
            required=True,
            example="route_1"
        ),
        openapi.Parameter(
            'stop_id',
            openapi.IN_QUERY,
            description="Stop ID (optional)",
            type=openapi.TYPE_STRING,
            required=False,
            example="stop_123"
        ),
    ],
    responses={
        200: openapi.Response(
            description="Real-time information retrieved successfully",
            examples={
                "application/json": {
                    "line_id": "route_1",
                    "stop_id": "stop_123",
                    "current_time": "14:30",
                    "status": "on_time",
                    "delay_minutes": 0,
                    "next_departures": [
                        {
                            "time": "14:35",
                            "delay_minutes": 0,
                            "status": "on_time",
                            "platform": "3",
                            "vehicle_id": "BUS1234",
                            "accessibility": {
                                "wheelchair_accessible": True,
                                "low_floor": True,
                                "ramp_available": True
                            }
                        }
                    ],
                    "service_status": "normal",
                    "last_updated": "2024-01-15T14:30:00",
                    "reliability": 95
                }
            }
        ),
        400: openapi.Response(
            description="Invalid parameters",
            examples={
                "application/json": {
                    "error": True,
                    "message": "line_id is required",
                    "code": "MISSING_LINE_ID"
                }
            }
        ),
        500: openapi.Response(
            description="Internal server error",
            examples={
                "application/json": {
                    "error": True,
                    "message": "Internal server error. Please try again later.",
                    "code": "INTERNAL_ERROR"
                }
            }
        )
    }
)
def real_time_info(request):
    """
    Get Västtrafik-style real-time information for bus lines.
    
    Parameters:
    - line_id: Bus line ID (required)
    - stop_id: Stop ID (optional)
    
    Returns:
    - JSON response with real-time information
    """
    try:
        line_id = request.GET.get('line_id')
        if not line_id:
            return Response({
                "error": True,
                "message": "line_id is required",
                "code": "MISSING_LINE_ID"
            }, status=400)
        
        stop_id = request.GET.get('stop_id')
        
        service = BusRouteService()
        real_time_data = service.get_real_time_info(line_id, stop_id)
        
        return Response(real_time_data)
        
    except Exception as e:
        logger.error(f"Error in real_time_info: {e}")
        return Response({
            "error": True,
            "message": "Internal server error. Please try again later.",
            "code": "INTERNAL_ERROR"
        }, status=500)
