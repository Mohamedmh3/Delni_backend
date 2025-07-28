"""
Service layer for bus route calculations and MongoDB operations
"""
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from django.conf import settings
from django.core.cache import cache
from django.db import connection
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from .exceptions import DatabaseConnectionError, NoRouteFoundError
from .graph_routing import bfs_multi_leg, nearest_point_on_line, distance
import hashlib
import pickle
from datetime import timedelta
from pymongo import MongoClient
from geopy.distance import geodesic

logger = logging.getLogger(__name__)

class BusRouteService:
    def __init__(self):
        from .mongo import initialize_mongo
        
        try:
            self.client, self.db, self.collection = initialize_mongo()
        except Exception as e:
            raise DatabaseConnectionError(f"MongoDB connection not available: {e}")
        
        # Cache configuration
        self.cache_timeout = getattr(settings, 'CACHE_TIMEOUT', 300)  # 5 minutes default
        self.cache_enabled = getattr(settings, 'CACHE_ENABLED', True)
    
    def _generate_cache_key(self, prefix: str, **kwargs) -> str:
        """
        Generate a cache key based on parameters.
        
        Args:
            prefix: Cache key prefix
            **kwargs: Parameters to include in cache key
            
        Returns:
            Cache key string
        """
        # Sort kwargs for consistent cache keys
        sorted_params = sorted(kwargs.items())
        param_string = json.dumps(sorted_params, sort_keys=True)
        
        # Create hash for long parameter strings
        if len(param_string) > 100:
            param_hash = hashlib.md5(param_string.encode()).hexdigest()
            return f"{prefix}:{param_hash}"
        
        return f"{prefix}:{param_string}"
    
    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """
        Get data from cache.
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Cached data or None if not found
        """
        if not self.cache_enabled:
            return None
            
        try:
            cached_data = cache.get(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_data
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
        
        return None
    
    def _set_cached_data(self, cache_key: str, data: Any, timeout: Optional[int] = None) -> None:
        """
        Store data in cache.
        
        Args:
            cache_key: Cache key to store
            data: Data to cache
            timeout: Cache timeout in seconds (optional)
        """
        if not self.cache_enabled:
            return
            
        try:
            timeout = timeout or self.cache_timeout
            cache.set(cache_key, data, timeout)
            logger.debug(f"Cached data for key: {cache_key} (timeout: {timeout}s)")
        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
    
    def _invalidate_route_cache(self, pattern: str = "route:*") -> None:
        """
        Invalidate route-related cache entries.
        
        Args:
            pattern: Cache key pattern to invalidate
        """
        if not self.cache_enabled:
            return
            
        try:
            # Note: This is a simplified cache invalidation
            # In production, consider using Redis with pattern-based deletion
            logger.info(f"Cache invalidation requested for pattern: {pattern}")
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")

    def distance_between_points(self, p1: List[float], p2: List[float]) -> float:
        """
        Calculate distance between two points [lng, lat]
        Returns distance in meters
        """
        try:
            return geodesic((p1[1], p1[0]), (p2[1], p2[0])).meters
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return float('inf')
    
    def find_nearest_point(self, user_point: List[float], route_coords: List[List[float]]) -> Tuple[List[float], float]:
        """
        Find the nearest point on a route to the user's location
        Returns (nearest_point, distance)
        """
        min_dist = float("inf")
        nearest = None
        
        for point in route_coords:
            d = self.distance_between_points(user_point, point)
            if d < min_dist:
                min_dist = d
                nearest = point
        
        return nearest, min_dist
    
    def interpolate_points(self, p1: List[float], p2: List[float], num_points: int = 10) -> List[List[float]]:
        """
        Interpolate points between p1 and p2 [lng, lat]
        num_points: number of points to generate between each pair
        """
        try:
            lng1, lat1 = p1
            lng2, lat2 = p2
            points = []
            
            for i in range(num_points + 1):
                lng = lng1 + (lng2 - lng1) * i / num_points
                lat = lat1 + (lat2 - lat1) * i / num_points
                points.append([lng, lat])
            
            return points
        except Exception as e:
            logger.error(f"Error interpolating points: {e}")
            return [p1, p2]
    
    def interpolate_line(self, coords: List[List[float]], points_per_segment: int = 30) -> List[List[float]]:
        """
        Generate interpolated points between each pair of coordinates on the line
        """
        try:
            interpolated = []
            for i in range(len(coords) - 1):
                segment_points = self.interpolate_points(coords[i], coords[i+1], points_per_segment)
                # Avoid repeating the start point of each segment except the first time
                if i > 0:
                    segment_points = segment_points[1:]
                interpolated.extend(segment_points)
            return interpolated
        except Exception as e:
            logger.error(f"Error interpolating line: {e}")
            return coords
    
    def get_walking_recommendation(self, distance: float) -> str:
        """
        Get walking recommendation based on distance.
        
        Args:
            distance: Walking distance in meters
            
        Returns:
            str: Walking recommendation
        """
        if distance <= 300:
            return "walk"
        elif distance <= 800:
            return "walk (moderate)"
        elif distance <= 1200:
            return "walk (long)"
        else:
            return "taxi recommended"
    
    def generate_walking_directions(self, from_point: List[float], to_point: List[float], 
                                   route_coords: List[List[float]] = None) -> Dict[str, Any]:
        """
        Generate Västtrafik-style walking directions with turn-by-turn instructions.
        
        Args:
            from_point: Starting point [lng, lat]
            to_point: Destination point [lng, lat]
            route_coords: Optional route coordinates for path following
            
        Returns:
            Dict with walking directions
        """
        distance_m = self.distance_between_points(from_point, to_point)
        estimated_time_min = distance_m / 5000 * 60  # 5 km/h walking speed
        
        # Generate turn-by-turn instructions
        instructions = []
        if route_coords and len(route_coords) > 1:
            # Follow route path
            for i in range(len(route_coords) - 1):
                current = route_coords[i]
                next_point = route_coords[i + 1]
                
                # Calculate bearing for direction
                bearing = self._calculate_bearing(current, next_point)
                direction = self._bearing_to_direction(bearing)
                
                segment_distance = self.distance_between_points(current, next_point)
                
                instructions.append({
                    "type": "walk",
                    "direction": direction,
                    "distance_m": round(segment_distance, 0),
                    "coordinates": [current, next_point],
                    "instruction": f"Walk {direction} for {round(segment_distance)}m"
                })
        else:
            # Direct path
            bearing = self._calculate_bearing(from_point, to_point)
            direction = self._bearing_to_direction(bearing)
            
            instructions.append({
                "type": "walk",
                "direction": direction,
                "distance_m": round(distance_m, 0),
                "coordinates": [from_point, to_point],
                "instruction": f"Walk {direction} for {round(distance_m)}m"
            })
        
        return {
            "total_distance_m": round(distance_m, 0),
            "estimated_time_min": round(estimated_time_min, 1),
            "instructions": instructions,
            "accessibility": {
                "wheelchair_accessible": True,  # Default assumption
                "elevator_available": False,
                "stairs_count": 0,
                "difficulty": "easy" if distance_m <= 500 else "moderate" if distance_m <= 1000 else "difficult"
            }
        }
    
    def _calculate_bearing(self, from_point: List[float], to_point: List[float]) -> float:
        """Calculate bearing between two points."""
        import math
        
        lat1 = math.radians(from_point[1])
        lat2 = math.radians(to_point[1])
        delta_lng = math.radians(to_point[0] - from_point[0])
        
        y = math.sin(delta_lng) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lng)
        
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360
    
    def _bearing_to_direction(self, bearing: float) -> str:
        """Convert bearing to cardinal direction."""
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = round(bearing / 45) % 8
        return directions[index]
    
    def calculate_route_length_m(self, coords):
        """
        Calculate the total length of a route in meters.
        """
        if not coords or len(coords) < 2:
            return 0.0
        total = 0.0
        for i in range(len(coords) - 1):
            total += self.distance_between_points(coords[i], coords[i+1])
        return total

    def find_routes(self, user_location: List[float], destination: List[float], 
                   preference: str = "closest", max_routes: int = 3) -> List[Dict]:
        """
        Find possible bus routes between user location and destination with caching
        
        Args:
            user_location: [lng, lat] of user's current location
            destination: [lng, lat] of user's destination
            preference: route selection preference ("closest", "fewest_walk", "fastest")
            max_routes: maximum number of routes to return
            
        Returns:
            List of route dictionaries with details
        """
        # Generate cache key
        cache_key = self._generate_cache_key(
            "route_find",
            user_location=user_location,
            destination=destination,
            preference=preference,
            max_routes=max_routes
        )
        
        # Try to get from cache first
        cached_routes = self._get_cached_data(cache_key)
        if cached_routes:
            logger.info(f"Returning cached routes for {user_location} to {destination}")
            return cached_routes
        
        try:
            routes = list(self.collection.find({}))
            possible_routes = []
            
            for route in routes:
                try:
                    coords = route.get("route", {}).get("coordinates", [])
                    if not coords:
                        continue
                    
                    # Interpolate points between coordinates for better line representation
                    detailed_coords = self.interpolate_line(coords, points_per_segment=20)
                    
                    entry_point, dist_to_entry = self.find_nearest_point(user_location, detailed_coords)
                    exit_point, dist_to_exit = self.find_nearest_point(destination, detailed_coords)
                    
                    # Check if both points are within reasonable distance from the route
                    if dist_to_entry < 600 and dist_to_exit < 1000:  # Approximate thresholds
                        walk_or_taxi = self.get_walking_recommendation(dist_to_exit)
                        
                        # Calculate total route length in meters
                        route_distance_m = self.calculate_route_length_m(coords)

                        # Build GeoJSON feature for the route
                        geojson_feature = {
                            "type": "Feature",
                            "geometry": route.get("route", {}),
                            "properties": {
                                "line_id": route.get("line_id", "Unknown"),
                                "route_name": route.get("route_name") or route.get("name", "Unknown Route"),
                                "route_name_en": route.get("name_en", None),
                                "type": route.get("type", None),
                                "direction": route.get("direction", None),
                            }
                        }

                        # Compose route info with all relevant metadata
                        route_info = {
                            "line_id": route.get("line_id", "Unknown"),
                            "route_name": route.get("route_name") or route.get("name", "Unknown Route"),
                            "route_name_en": route.get("name_en", None),
                            "type": route.get("type", None),
                            "direction": route.get("direction", None),
                            "entry_point": entry_point,
                            "exit_point": exit_point,
                            "distance_to_entry": round(dist_to_entry, 2),
                            "distance_to_exit": round(dist_to_exit, 2),
                            "total_walking_distance": round(dist_to_entry + dist_to_exit, 2),
                            "walk_or_taxi": walk_or_taxi,
                            "coordinates": coords,
                            "route_length": len(coords),
                            "route_distance_m": round(route_distance_m, 2),
                            "geojson": geojson_feature
                        }

                        # Optionally include all other fields from the document
                        for key in route:
                            if key not in route_info and key not in ["route"]:
                                route_info[key] = route[key]
                        
                        possible_routes.append(route_info)
                
                except Exception as e:
                    logger.error(f"Error processing route {route.get('line_id', 'Unknown')}: {e}")
                    continue
            
            # Sort routes based on preference
            if preference == "closest":
                possible_routes.sort(key=lambda r: r["distance_to_entry"])
            elif preference == "fewest_walk":
                possible_routes.sort(key=lambda r: r["total_walking_distance"])
            elif preference == "fastest":
                possible_routes.sort(key=lambda r: r["route_distance_m"])
            
            result = possible_routes[:max_routes]
            
            # Cache the results
            self._set_cached_data(cache_key, result, timeout=600)  # Cache for 10 minutes
            
            return result
            
        except Exception as e:
            logger.error(f"Error finding routes: {e}")
            return []
    
    def get_route_statistics(self) -> Dict:
        """
        Get basic statistics about available routes
        """
        try:
            total_routes = self.collection.count_documents({})
            return {
                "total_routes": total_routes,
                "database": settings.MONGODB_DATABASE,
                "collection": settings.MONGODB_COLLECTION
            }
        except Exception as e:
            logger.error(f"Error getting route statistics: {e}")
            return {"error": "Failed to get statistics"}
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close() 

    def get_all_routes(self, projection=None):
        """
        Get all routes with optional projection for performance optimization.
        
        Args:
            projection: MongoDB projection dict to fetch only needed fields
            
        Returns:
            List of route documents
        """
        if projection is None:
            # Default projection for route finding - only fetch essential fields
            projection = {
                'line_id': 1,
                'name': 1,
                'type': 1,
                'direction': 1,
                'route.coordinates': 1,
                '_id': 0  # Exclude MongoDB _id for better performance
            }
        
        return list(self.collection.find({}, projection))
    
    def get_routes_by_type(self, route_type, projection=None):
        """
        Get routes filtered by type with optional projection.
        
        Args:
            route_type: Type of route (bus, microbus, etc.)
            projection: MongoDB projection dict
            
        Returns:
            List of route documents
        """
        if projection is None:
            projection = {
                'line_id': 1,
                'name': 1,
                'route.coordinates': 1,
                '_id': 0
            }
        
        return list(self.collection.find({'type': route_type}, projection))
    
    def get_routes_in_bounds(self, min_lng, max_lng, min_lat, max_lat, projection=None):
        """
        Get routes within geographic bounds for performance optimization.
        
        Args:
            min_lng, max_lng, min_lat, max_lat: Geographic bounds
            projection: MongoDB projection dict
            
        Returns:
            List of route documents within bounds
        """
        if projection is None:
            projection = {
                'line_id': 1,
                'name': 1,
                'route.coordinates': 1,
                '_id': 0
            }
        
        # MongoDB geospatial query for routes within bounds
        query = {
            'route.coordinates': {
                '$geoWithin': {
                    '$box': [
                        [min_lng, min_lat],
                        [max_lng, max_lat]
                    ]
                }
            }
        }
        
        return list(self.collection.find(query, projection))
    
    def get_route_by_id(self, line_id, projection=None):
        """
        Get a specific route by line_id with optional projection.
        
        Args:
            line_id: The route line ID
            projection: MongoDB projection dict
            
        Returns:
            Route document or None
        """
        if projection is None:
            projection = {
                'line_id': 1,
                'name': 1,
                'type': 1,
                'direction': 1,
                'route.coordinates': 1,
                '_id': 0
            }
        
        return self.collection.find_one({'line_id': line_id}, projection)
    
    def get_routes_for_routing(self, origin, destination, max_distance_km=50):
        """
        Get routes optimized for routing between two points.
        Only fetches routes that could potentially be used.
        
        Args:
            origin: [lng, lat] of origin
            destination: [lng, lat] of destination
            max_distance_km: Maximum distance to consider for route search
            
        Returns:
            List of relevant route documents
        """
        # Calculate bounding box around origin and destination
        buffer_degrees = max_distance_km / 111.0  # Approximate km to degrees
        
        min_lng = min(origin[0], destination[0]) - buffer_degrees
        max_lng = max(origin[0], destination[0]) + buffer_degrees
        min_lat = min(origin[1], destination[1]) - buffer_degrees
        max_lat = max(origin[1], destination[1]) + buffer_degrees
        
        # Projection for routing - only essential fields
        projection = {
            'line_id': 1,
            'name': 1,
            'type': 1,
            'direction': 1,
            'route.coordinates': 1,
            '_id': 0
        }
        
        return self.get_routes_in_bounds(min_lng, max_lng, min_lat, max_lat, projection)
    
    def create_indexes(self):
        """
        Create MongoDB indexes for optimal performance.
        Should be called during application startup or data migration.
        """
        try:
            # Index on line_id for fast lookups
            self.collection.create_index('line_id', unique=True)
            
            # Index on route type for filtering
            self.collection.create_index('type')
            
            # Geospatial index on route coordinates for spatial queries
            self.collection.create_index([('route.coordinates', '2dsphere')])
            
            # Compound index for common queries
            self.collection.create_index([('type', 1), ('direction', 1)])
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")
    
    def get_database_stats(self):
        """
        Get database statistics for monitoring and optimization.
        
        Returns:
            Dict with database statistics
        """
        try:
            stats = {
                'total_routes': self.collection.count_documents({}),
                'routes_by_type': {},
                'database_name': self.db.name,
                'collection_name': self.collection.name
            }
            
            # Count routes by type
            pipeline = [
                {'$group': {'_id': '$type', 'count': {'$sum': 1}}},
                {'$sort': {'count': -1}}
            ]
            
            for result in self.collection.aggregate(pipeline):
                stats['routes_by_type'][result['_id']] = result['count']
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)} 

    def get_real_time_info(self, line_id: str, stop_id: str = None) -> Dict[str, Any]:
        """
        Get Västtrafik-style real-time information for a bus line.
        
        Args:
            line_id: Bus line ID
            stop_id: Optional stop ID for specific stop info
            
        Returns:
            Dict with real-time information
        """
        # This is a placeholder - in a real implementation, this would connect to real-time APIs
        # For now, we'll simulate Västtrafik-style real-time data
        
        import random
        from datetime import datetime, timedelta
        
        current_time = datetime.now()
        
        # Simulate real-time delays and status
        base_delay = random.randint(-2, 5)  # -2 to +5 minutes
        status = "on_time" if base_delay <= 1 else "delayed" if base_delay > 1 else "early"
        
        # Generate next departures
        next_departures = []
        for i in range(3):  # Next 3 departures
            departure_time = current_time + timedelta(minutes=10 + i * 15 + base_delay)
            next_departures.append({
                "time": departure_time.strftime("%H:%M"),
                "delay_minutes": base_delay if i == 0 else 0,
                "status": status if i == 0 else "scheduled",
                "platform": f"{random.randint(1, 8)}",
                "vehicle_id": f"BUS{random.randint(1000, 9999)}",
                "accessibility": {
                    "wheelchair_accessible": random.choice([True, True, True, False]),  # 75% accessible
                    "low_floor": random.choice([True, True, False]),
                    "ramp_available": random.choice([True, False])
                }
            })
        
        return {
            "line_id": line_id,
            "stop_id": stop_id,
            "current_time": current_time.strftime("%H:%M"),
            "status": status,
            "delay_minutes": base_delay,
            "next_departures": next_departures,
            "service_status": "normal",  # normal, reduced, cancelled
            "last_updated": current_time.isoformat(),
            "reliability": random.randint(85, 98)  # Percentage
        }
    
    def get_line_frequency(self, line_id: str) -> Dict[str, Any]:
        """
        Get Västtrafik-style frequency information for a bus line.
        
        Args:
            line_id: Bus line ID
            
        Returns:
            Dict with frequency information
        """
        # Simulate frequency data
        frequencies = {
            "weekday": {
                "peak_hours": "Every 5-10 minutes",
                "off_peak": "Every 15-20 minutes",
                "evening": "Every 30 minutes",
                "night": "Every 60 minutes"
            },
            "weekend": {
                "day": "Every 20-30 minutes",
                "evening": "Every 45 minutes",
                "night": "Every 90 minutes"
            }
        }
        
        return {
            "line_id": line_id,
            "frequencies": frequencies,
            "operating_hours": {
                "weekday": "05:00-24:00",
                "weekend": "06:00-23:00"
            },
            "service_days": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        } 