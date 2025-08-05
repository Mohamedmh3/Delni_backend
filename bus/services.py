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
import math

logger = logging.getLogger(__name__)

class CoordinateSimplifier:
    """
    Utility class for coordinate simplification using Ramer-Douglas-Peucker (RDP) algorithm
    """
    @staticmethod
    def perpendicular_distance(point, line_start, line_end):
        """Calculate perpendicular distance from point to line segment"""
        if line_start == line_end:
            return distance(point, line_start)
        
        # Vector from line_start to line_end
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        
        # Vector from line_start to point
        px = point[0] - line_start[0]
        py = point[1] - line_start[1]
        
        # Projection of point onto line
        t = max(0, min(1, (px * dx + py * dy) / (dx * dx + dy * dy)))
        
        # Closest point on line segment
        closest_x = line_start[0] + t * dx
        closest_y = line_start[1] + t * dy
        
        return distance(point, [closest_x, closest_y])
    
    @staticmethod
    def rdp_simplification(coordinates, epsilon):
        """
        Ramer-Douglas-Peucker algorithm for coordinate simplification
        
        Args:
            coordinates: List of [lng, lat] coordinates
            epsilon: Simplification tolerance in meters
            
        Returns:
            Simplified coordinate list with preserved geometry
        """
        if len(coordinates) <= 2:
            return coordinates
        
        # Find point with maximum distance
        max_distance = 0
        max_index = 0
        
        for i in range(1, len(coordinates) - 1):
            dist = CoordinateSimplifier.perpendicular_distance(
                coordinates[i], coordinates[0], coordinates[-1]
            )
            if dist > max_distance:
                max_distance = dist
                max_index = i
        
        # If max distance is greater than epsilon, recursively simplify
        if max_distance > epsilon:
            # Recursive call
            rec_results1 = CoordinateSimplifier.rdp_simplification(
                coordinates[:max_index + 1], epsilon
            )
            rec_results2 = CoordinateSimplifier.rdp_simplification(
                coordinates[max_index:], epsilon
            )
            
            # Combine results (avoid duplicate point)
            return rec_results1[:-1] + rec_results2
        else:
            # All points in between can be eliminated
            return [coordinates[0], coordinates[-1]]
    
    @staticmethod
    def douglas_peucker(coordinates, epsilon):
        """
        Alias for RDP simplification (backward compatibility)
        """
        return CoordinateSimplifier.rdp_simplification(coordinates, epsilon)
    
    @staticmethod
    def douglas_peucker(coordinates, epsilon):
        """
        Simplify coordinates using Douglas-Peucker algorithm
        
        Args:
            coordinates: List of [lng, lat] coordinates
            epsilon: Simplification tolerance in meters
            
        Returns:
            Simplified coordinate list
        """
        if len(coordinates) <= 2:
            return coordinates
        
        # Find point with maximum distance
        max_distance = 0
        max_index = 0
        
        for i in range(1, len(coordinates) - 1):
            dist = CoordinateSimplifier.perpendicular_distance(
                coordinates[i], coordinates[0], coordinates[-1]
            )
            if dist > max_distance:
                max_distance = dist
                max_index = i
        
        # If max distance is greater than epsilon, recursively simplify
        if max_distance > epsilon:
            # Recursive call
            rec_results1 = CoordinateSimplifier.douglas_peucker(
                coordinates[:max_index + 1], epsilon
            )
            rec_results2 = CoordinateSimplifier.douglas_peucker(
                coordinates[max_index:], epsilon
            )
            
            # Combine results (avoid duplicate point)
            return rec_results1[:-1] + rec_results2
        else:
            # All points in between can be eliminated
            return [coordinates[0], coordinates[-1]]
    
    @staticmethod
    def adaptive_simplification(coordinates, target_points=100):
        """
        Adaptively simplify coordinates to target number of points
        
        Args:
            coordinates: List of [lng, lat] coordinates
            target_points: Target number of points after simplification
            
        Returns:
            Simplified coordinate list
        """
        if len(coordinates) <= target_points:
            return coordinates
        
        # Calculate initial epsilon based on route length
        total_length = sum(
            distance(coordinates[i], coordinates[i+1]) 
            for i in range(len(coordinates)-1)
        )
        
        # Start with a reasonable epsilon
        epsilon = total_length / (len(coordinates) * 10)
        
        # Binary search for optimal epsilon
        min_epsilon = 0
        max_epsilon = total_length
        
        best_result = coordinates
        for _ in range(10):  # Max 10 iterations
            simplified = CoordinateSimplifier.douglas_peucker(coordinates, epsilon)
            
            if len(simplified) <= target_points:
                best_result = simplified
                max_epsilon = epsilon
                epsilon = (min_epsilon + epsilon) / 2
            else:
                min_epsilon = epsilon
                epsilon = (epsilon + max_epsilon) / 2
        
        return best_result

class GraphBasedRouter:
    """
    Advanced graph-based routing with entry/exit point preprocessing
    """
    def __init__(self, service):
        self.service = service
        self.graph_cache = {}
        self.entry_exit_cache = {}
        
    def preprocess_entry_exit_points(self, routes):
        """
        Preprocess and store entry/exit points for each route
        
        Args:
            routes: List of route documents
            
        Returns:
            Dict with preprocessed entry/exit points
        """
        processed_routes = {}
        
        for route in routes:
            line_id = route['line_id']
            coords = route['route']['coordinates']
            
            # Generate entry/exit points every 500m along the route
            entry_points = self._generate_entry_exit_points(coords, interval_m=500)
            
            processed_routes[line_id] = {
                'route': route,
                'entry_points': entry_points,
                'simplified_coords': self.service.optimize_coordinates_for_routing(coords)
            }
            
            # Cache for future use
            self.entry_exit_cache[line_id] = entry_points
        
        return processed_routes
    
    def _generate_entry_exit_points(self, coordinates, interval_m=500):
        """
        Generate entry/exit points along a route at regular intervals
        
        Args:
            coordinates: Route coordinates
            interval_m: Interval in meters between points
            
        Returns:
            List of entry/exit points with metadata
        """
        entry_points = []
        cumulative_distance = 0
        
        for i in range(len(coordinates) - 1):
            segment_distance = distance(coordinates[i], coordinates[i + 1])
            
            # Check if we need to add an entry point
            while cumulative_distance + segment_distance >= len(entry_points) * interval_m:
                # Calculate position along this segment
                target_distance = len(entry_points) * interval_m - cumulative_distance
                ratio = target_distance / segment_distance
                
                # Interpolate point
                lng = coordinates[i][0] + ratio * (coordinates[i + 1][0] - coordinates[i][0])
                lat = coordinates[i][1] + ratio * (coordinates[i + 1][1] - coordinates[i][1])
                
                entry_points.append({
                    'point': [lng, lat],
                    'index': i,
                    'distance_from_start': len(entry_points) * interval_m,
                    'segment_ratio': ratio
                })
            
            cumulative_distance += segment_distance
        
        return entry_points
    
    def build_transfer_graph(self, routes, transfer_threshold_m=400):
        """
        Build in-memory graph for efficient transfer calculations
        
        Args:
            routes: Preprocessed routes with entry/exit points
            transfer_threshold_m: Maximum distance for transfers
            
        Returns:
            Graph adjacency list
        """
        graph = {}
        
        for line_id, route_data in routes.items():
            graph[line_id] = {
                'entry_points': route_data['entry_points'],
                'transfers': []
            }
            
            # Find transfers to other routes
            for other_line_id, other_route_data in routes.items():
                if line_id != other_line_id:
                    transfers = self._find_transfers(
                        route_data['entry_points'],
                        other_route_data['entry_points'],
                        transfer_threshold_m
                    )
                    
                    if transfers:
                        graph[line_id]['transfers'].extend(transfers)
        
        return graph
    
    def _find_transfers(self, entry_points_1, entry_points_2, threshold_m):
        """
        Find transfer points between two routes
        
        Args:
            entry_points_1: Entry points of first route
            entry_points_2: Entry points of second route
            threshold_m: Maximum transfer distance
            
        Returns:
            List of transfer points
        """
        transfers = []
        
        for ep1 in entry_points_1:
            for ep2 in entry_points_2:
                dist = distance(ep1['point'], ep2['point'])
                if dist <= threshold_m:
                    transfers.append({
                        'from_point': ep1,
                        'to_point': ep2,
                        'distance': dist,
                        'target_route': ep2.get('route_id')
                    })
        
        return transfers
    
    def find_optimal_routes(self, origin, destination, routes, max_legs=4):
        """
        Find optimal routes using graph-based BFS (legacy method)
        
        Args:
            origin: [lng, lat] of origin
            destination: [lng, lat] of destination
            routes: Preprocessed routes
            max_legs: Maximum number of route legs
            
        Returns:
            List of optimal routes
        """
        return self.find_all_possible_routes(origin, destination, routes, max_legs)
    
    def find_all_possible_routes(self, origin, destination, routes, max_legs=4):
        """
        Find all possible routes using comprehensive BFS (no walking distance limits)
        
        Args:
            origin: [lng, lat] of origin
            destination: [lng, lat] of destination
            routes: Preprocessed routes
            max_legs: Maximum number of route legs
            
        Returns:
            List of all possible routes (up to 50 for comprehensive exploration)
        """
        # Preprocess routes if not already done
        if not self.entry_exit_cache:
            routes = self.preprocess_entry_exit_points(routes)
        
        # Build transfer graph
        graph = self.build_transfer_graph(routes)
        
        # Find ALL routes near origin (no distance limit)
        origin_routes = self._find_all_routes_near_point(origin, routes)
        
        # Find ALL routes near destination (no distance limit)
        dest_routes = self._find_all_routes_near_point(destination, routes)
        
        # Perform comprehensive BFS to find all possible routes
        all_routes = self._comprehensive_bfs_routing(
            origin, destination, origin_routes, dest_routes, graph, max_legs
        )
        
        return all_routes
    
    def _find_routes_near_point(self, point, routes, max_distance_m=500):
        """
        Find routes and entry/exit points near a given point
        
        Args:
            point: [lng, lat] coordinates
            routes: Preprocessed routes
            max_distance_m: Maximum distance to consider
            
        Returns:
            List of nearby routes with entry/exit points
        """
        nearby_routes = []
        
        for line_id, route_data in routes.items():
            coords = route_data['route']['route']['coordinates']
            
            # Find nearest point on route
            nearest_point, nearest_dist, nearest_index = nearest_point_on_line(point, coords)
            
            if nearest_dist <= max_distance_m:
                nearby_routes.append({
                    'line_id': line_id,
                    'route_data': route_data,
                    'entry_point': {
                        'point': nearest_point,
                        'distance': nearest_dist,
                        'index': nearest_index
                    }
                })
        
        return nearby_routes
    
    def _find_all_routes_near_point(self, point, routes):
        """
        Find ALL routes near a given point (no distance limit)
        
        Args:
            point: [lng, lat] coordinates
            routes: Preprocessed routes
            
        Returns:
            List of all nearby routes with entry/exit points
        """
        nearby_routes = []
        
        for line_id, route_data in routes.items():
            coords = route_data['route']['route']['coordinates']
            
            # Find nearest point on route (no distance limit)
            nearest_point, nearest_dist, nearest_index = nearest_point_on_line(point, coords)
            
            # Include ALL routes regardless of distance
            nearby_routes.append({
                'line_id': line_id,
                'route_data': route_data,
                'entry_point': {
                    'point': nearest_point,
                    'distance': nearest_dist,
                    'index': nearest_index
                }
            })
        
        return nearby_routes
    
    def _comprehensive_bfs_routing(self, origin, destination, origin_routes, dest_routes, graph, max_legs):
        """
        Comprehensive BFS for finding all possible routes (no walking distance limits)
        
        Args:
            origin: Origin coordinates
            destination: Destination coordinates
            origin_routes: All routes near origin
            dest_routes: All routes near destination
            graph: Transfer graph
            max_legs: Maximum route legs
            
        Returns:
            List of all possible routes (up to 50)
        """
        from collections import deque
        
        all_routes = []
        visited = set()
        
        # Start BFS from each origin route
        for origin_route in origin_routes:
            queue = deque([{
                'route_sequence': [origin_route['line_id']],
                'current_route': origin_route['line_id'],
                'current_point': origin_route['entry_point']['point'],
                'total_distance': origin_route['entry_point']['distance'],
                'legs': 1,
                'visited_routes': set([origin_route['line_id']])
            }])
            
            while queue and len(all_routes) < 50:  # Increased limit for comprehensive search
                current = queue.popleft()
                
                # Check if we can reach destination from current route
                if current['current_route'] in [r['line_id'] for r in dest_routes]:
                    # Calculate final route
                    final_route = self._construct_comprehensive_route(
                        current, destination, origin, dest_routes
                    )
                    all_routes.append(final_route)
                    continue
                
                # Explore transfers (no distance limits)
                if current['legs'] < max_legs:
                    for transfer in graph[current['current_route']]['transfers']:
                        transfer_key = f"{current['current_route']}_{transfer['target_route']}"
                        
                        if transfer_key not in visited:
                            visited.add(transfer_key)
                            
                            new_state = {
                                'route_sequence': current['route_sequence'] + [transfer['target_route']],
                                'current_route': transfer['target_route'],
                                'current_point': transfer['to_point']['point'],
                                'total_distance': current['total_distance'] + transfer['distance'],
                                'legs': current['legs'] + 1,
                                'visited_routes': current['visited_routes'].copy()
                            }
                            new_state['visited_routes'].add(transfer['target_route'])
                            
                            queue.append(new_state)
        
        return all_routes
    
    def _construct_comprehensive_route(self, bfs_state, destination, origin, dest_routes):
        """
        Construct comprehensive route from BFS state (includes all metrics)
        
        Args:
            bfs_state: BFS traversal state
            destination: Destination coordinates
            origin: Origin coordinates
            dest_routes: Routes near destination
            
        Returns:
            Comprehensive route with all metrics
        """
        # Find destination route
        dest_route = next(r for r in dest_routes if r['line_id'] == bfs_state['current_route'])
        
        total_walking = bfs_state['total_distance'] + dest_route['entry_point']['distance']
        
        return {
            'route_sequence': bfs_state['route_sequence'],
            'total_distance': bfs_state['total_distance'] + dest_route['entry_point']['distance'],
            'legs': bfs_state['legs'],
            'origin_walk': bfs_state['total_distance'],
            'destination_walk': dest_route['entry_point']['distance'],
            'total_walking': total_walking,
            'transfers': bfs_state['legs'] - 1,
            'requires_taxi_suggestion': total_walking > 500  # Flag for taxi suggestion
        }
    
    def _bfs_routing(self, origin, destination, origin_routes, dest_routes, graph, max_legs):
        """
        Breadth-First Search for optimal routing
        
        Args:
            origin: Origin coordinates
            destination: Destination coordinates
            origin_routes: Routes near origin
            dest_routes: Routes near destination
            graph: Transfer graph
            max_legs: Maximum route legs
            
        Returns:
            List of optimal routes
        """
        from collections import deque
        
        optimal_routes = []
        visited = set()
        
        # Start BFS from each origin route
        for origin_route in origin_routes:
            queue = deque([{
                'route_sequence': [origin_route['line_id']],
                'current_route': origin_route['line_id'],
                'current_point': origin_route['entry_point']['point'],
                'total_distance': origin_route['entry_point']['distance'],
                'legs': 1
            }])
            
            while queue and len(optimal_routes) < 10:  # Limit to 10 routes
                current = queue.popleft()
                
                # Check if we can reach destination from current route
                if current['current_route'] in [r['line_id'] for r in dest_routes]:
                    # Calculate final route
                    final_route = self._construct_route(
                        current, destination, origin, dest_routes
                    )
                    optimal_routes.append(final_route)
                    continue
                
                # Explore transfers
                if current['legs'] < max_legs:
                    for transfer in graph[current['current_route']]['transfers']:
                        transfer_key = f"{current['current_route']}_{transfer['target_route']}"
                        
                        if transfer_key not in visited:
                            visited.add(transfer_key)
                            
                            new_state = {
                                'route_sequence': current['route_sequence'] + [transfer['target_route']],
                                'current_route': transfer['target_route'],
                                'current_point': transfer['to_point']['point'],
                                'total_distance': current['total_distance'] + transfer['distance'],
                                'legs': current['legs'] + 1
                            }
                            
                            queue.append(new_state)
        
        # Sort by total distance and return top routes
        optimal_routes.sort(key=lambda x: x['total_distance'])
        return optimal_routes[:10]
    
    def _construct_route(self, bfs_state, destination, origin, dest_routes):
        """
        Construct final route from BFS state
        
        Args:
            bfs_state: BFS traversal state
            destination: Destination coordinates
            origin: Origin coordinates
            dest_routes: Routes near destination
            
        Returns:
            Constructed route
        """
        # Find destination route
        dest_route = next(r for r in dest_routes if r['line_id'] == bfs_state['current_route'])
        
        return {
            'route_sequence': bfs_state['route_sequence'],
            'total_distance': bfs_state['total_distance'] + dest_route['entry_point']['distance'],
            'legs': bfs_state['legs'],
            'origin_walk': bfs_state['total_distance'],
            'destination_walk': dest_route['entry_point']['distance'],
            'transfers': bfs_state['legs'] - 1
        }

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
        
        # Coordinate simplification settings
        self.simplification_enabled = getattr(settings, 'COORDINATE_SIMPLIFICATION_ENABLED', True)
        self.target_coordinates_per_route = getattr(settings, 'TARGET_COORDINATES_PER_ROUTE', 100)
        self.simplification_tolerance_m = getattr(settings, 'SIMPLIFICATION_TOLERANCE_M', 50)
        
        # Geospatial query optimization settings
        self.use_bounding_box_filter = getattr(settings, 'USE_BOUNDING_BOX_FILTER', True)
        self.bounding_box_buffer_km = getattr(settings, 'BOUNDING_BOX_BUFFER_KM', 5)
        self.max_routes_per_query = getattr(settings, 'MAX_ROUTES_PER_QUERY', 50)
        
        # Graph-based routing settings
        self.transfer_threshold_m = getattr(settings, 'TRANSFER_THRESHOLD_M', 400)
        self.entry_exit_interval_m = getattr(settings, 'ENTRY_EXIT_INTERVAL_M', 500)
        self.max_route_legs = getattr(settings, 'MAX_ROUTE_LEGS', 4)
        
        # Initialize graph-based router
        self.graph_router = GraphBasedRouter(self)
    
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
    
    def cache_route_data(self, route_id: str, route_data: Dict[str, Any], timeout: int = 3600) -> None:
        """
        Cache route data with optimized storage.
        
        Args:
            route_id: Unique route identifier
            route_data: Route data to cache
            timeout: Cache timeout in seconds
        """
        if not self.cache_enabled:
            return
            
        try:
            # Create separate cache entries for different data types
            cache_key_base = f"route:{route_id}"
            
            # Cache basic route info (longer timeout)
            basic_info = {
                'line_id': route_data.get('line_id'),
                'name': route_data.get('name'),
                'type': route_data.get('type'),
                'direction': route_data.get('direction')
            }
            self._set_cached_data(f"{cache_key_base}:basic", basic_info, timeout * 2)
            
            # Cache coordinates separately (shorter timeout, larger data)
            if 'route' in route_data and 'coordinates' in route_data['route']:
                coords = route_data['route']['coordinates']
                # Compress coordinates for storage
                compressed_coords = self._compress_coordinates(coords)
                self._set_cached_data(f"{cache_key_base}:coords", compressed_coords, timeout)
            
            logger.debug(f"Cached route data for {route_id}")
            
        except Exception as e:
            logger.warning(f"Error caching route data: {e}")
    
    def get_cached_route_data(self, route_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached route data with optimized retrieval.
        
        Args:
            route_id: Unique route identifier
            
        Returns:
            Cached route data or None
        """
        if not self.cache_enabled:
            return None
            
        try:
            cache_key_base = f"route:{route_id}"
            
            # Get basic info
            basic_info = self._get_cached_data(f"{cache_key_base}:basic")
            if not basic_info:
                return None
            
            # Get coordinates
            compressed_coords = self._get_cached_data(f"{cache_key_base}:coords")
            if compressed_coords:
                coords = self._decompress_coordinates(compressed_coords)
                basic_info['route'] = {'coordinates': coords}
            
            return basic_info
            
        except Exception as e:
            logger.warning(f"Error retrieving cached route data: {e}")
            return None
    
    def _compress_coordinates(self, coordinates: List[List[float]]) -> str:
        """
        Compress coordinates for efficient caching.
        
        Args:
            coordinates: List of [lng, lat] coordinates
            
        Returns:
            Compressed coordinate string
        """
        try:
            # Convert to bytes and compress
            import gzip
            import json
            
            coord_str = json.dumps(coordinates)
            compressed = gzip.compress(coord_str.encode('utf-8'))
            return compressed.hex()
            
        except Exception as e:
            logger.warning(f"Error compressing coordinates: {e}")
            return json.dumps(coordinates)
    
    def _decompress_coordinates(self, compressed_data: str) -> List[List[float]]:
        """
        Decompress coordinates from cache.
        
        Args:
            compressed_data: Compressed coordinate string
            
        Returns:
            Decompressed coordinate list
        """
        try:
            import gzip
            import json
            
            # Try to decompress
            try:
                compressed_bytes = bytes.fromhex(compressed_data)
                decompressed = gzip.decompress(compressed_bytes)
                return json.loads(decompressed.decode('utf-8'))
            except:
                # Fallback to direct JSON parsing
                return json.loads(compressed_data)
                
        except Exception as e:
            logger.warning(f"Error decompressing coordinates: {e}")
            return []
    
    def cache_geospatial_query(self, query_params: Dict[str, Any], results: List[Dict[str, Any]], timeout: int = 1800) -> None:
        """
        Cache geospatial query results.
        
        Args:
            query_params: Query parameters
            results: Query results
            timeout: Cache timeout in seconds
        """
        if not self.cache_enabled:
            return
            
        try:
            # Create cache key from query parameters
            cache_key = self._generate_cache_key("geospatial_query", **query_params)
            
            # Cache results with shorter timeout for geospatial queries
            self._set_cached_data(cache_key, results, timeout)
            
            logger.debug(f"Cached geospatial query with {len(results)} results")
            
        except Exception as e:
            logger.warning(f"Error caching geospatial query: {e}")
    
    def get_cached_geospatial_query(self, query_params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached geospatial query results.
        
        Args:
            query_params: Query parameters
            
        Returns:
            Cached results or None
        """
        if not self.cache_enabled:
            return None
            
        try:
            cache_key = self._generate_cache_key("geospatial_query", **query_params)
            return self._get_cached_data(cache_key)
            
        except Exception as e:
            logger.warning(f"Error retrieving cached geospatial query: {e}")
            return None

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
                   preference: str = "comprehensive", max_routes: int = 10) -> List[Dict]:
        """
        Find comprehensive routes between two points using graph-based BFS (no walking distance limits).
        
        Args:
            user_location: [lng, lat] of user's current location
            destination: [lng, lat] of user's destination
            preference: route selection preference ("comprehensive", "fastest", "fewest_transfers", "fewest_walk")
            max_routes: maximum number of routes to return (default 10)
            
        Returns:
            List of route dictionaries sorted by: fewest transfers → fastest time → least walking
            Includes taxi suggestions for routes with >500m walking distance
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
            # Use bounding box filtering with 500m buffer for efficient querying
            routes = self.get_routes_for_routing(user_location, destination)
            
            if not routes:
                logger.warning(f"No routes found between {user_location} and {destination}")
                return []
            
            # Use graph-based routing for optimal results
            optimal_routes = self.find_routes_with_graph_based_bfs(
                user_location, destination, routes, preference, max_routes
            )
            
            # Cache the results
            self._set_cached_data(cache_key, optimal_routes, self.cache_timeout)
            
            return optimal_routes
            
        except Exception as e:
            logger.error(f"Error finding routes: {e}")
            return []
    
    def find_routes_with_graph_based_bfs(self, origin, destination, routes, preference, max_routes):
        """
        Find routes using graph-based BFS with comprehensive exploration (no walking distance limits)
        
        Args:
            origin: [lng, lat] of origin
            destination: [lng, lat] of destination
            routes: List of route documents
            preference: Route preference
            max_routes: Maximum routes to return (default 10)
            
        Returns:
            List of optimal routes sorted by: fewest transfers → fastest time → least walking
        """
        try:
            # Preprocess routes with entry/exit points (no distance filtering)
            processed_routes = self.graph_router.preprocess_entry_exit_points(routes)
            
            # Find all possible routes using comprehensive BFS (no walking distance limits)
            all_routes = self.graph_router.find_all_possible_routes(
                origin, destination, processed_routes, self.max_route_legs
            )
            
            # Apply comprehensive sorting: fewest transfers → fastest time → least walking
            sorted_routes = self._sort_routes_comprehensive(all_routes)
            
            # Process and format results (up to 10 routes)
            processed_results = []
            for route in sorted_routes[:max_routes]:
                processed_result = self._process_graph_route_result(
                    route, origin, destination, processed_routes
                )
                processed_results.append(processed_result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive graph-based BFS: {e}")
            # Fallback to existing BFS algorithm
            return self._fallback_to_existing_bfs(origin, destination, routes, max_routes)
    
    def _sort_routes_comprehensive(self, routes):
        """
        Sort routes comprehensively: fewest transfers → fastest time → least walking
        
        Args:
            routes: List of route results from BFS
            
        Returns:
            Sorted list of routes
        """
        # Calculate total time for each route
        for route in routes:
            route['total_time'] = self._calculate_route_time(route, [])
            route['total_walking'] = route.get('origin_walk', 0) + route.get('destination_walk', 0)
        
        # Sort by: fewest transfers → fastest time → least walking
        sorted_routes = sorted(routes, key=lambda x: (
            x.get('transfers', 0),           # Primary: fewest transfers
            x.get('total_time', float('inf')),  # Secondary: fastest time
            x.get('total_walking', float('inf'))  # Tertiary: least walking
        ))
        
        return sorted_routes
    
    def _process_graph_route_result(self, route_result, origin, destination, processed_routes):
        """
        Process graph-based route result into standardized format with taxi suggestions
        
        Args:
            route_result: Result from graph-based BFS
            origin: Origin coordinates
            destination: Destination coordinates
            processed_routes: Preprocessed routes
            
        Returns:
            Processed route result with taxi suggestions
        """
        try:
            # Extract route information
            route_sequence = route_result['route_sequence']
            legs = []
            
            for i, line_id in enumerate(route_sequence):
                if line_id in processed_routes:
                    route_data = processed_routes[line_id]['route']
                    
                    # Get coordinates for rendering (full coordinates)
                    coords = self.get_coordinates_for_use_case(route_data, 'rendering')
                    
                    leg = {
                        'line_id': line_id,
                        'name': route_data.get('name', 'Unknown Route'),
                        'type': route_data.get('type', 'bus'),
                        'direction': route_data.get('direction', 'unknown'),
                        'coordinates': coords,
                        'leg_index': i,
                        'is_transfer': i > 0
                    }
                    
                    legs.append(leg)
            
            # Calculate timing estimates
            total_time = self._calculate_route_time(route_result, legs)
            
            # Calculate total walking distance
            total_walking = route_result.get('total_walking', 
                route_result.get('origin_walk', 0) + route_result.get('destination_walk', 0))
            
            # Determine if taxi suggestion is needed
            requires_taxi_suggestion = total_walking > 500
            
            # Generate taxi suggestion message
            taxi_suggestion = self._generate_taxi_suggestion(total_walking) if requires_taxi_suggestion else None
            
            return {
                'route_sequence': route_sequence,
                'legs': legs,
                'total_distance': route_result['total_distance'],
                'total_time': total_time,
                'transfers': route_result['transfers'],
                'origin_walk': route_result['origin_walk'],
                'destination_walk': route_result['destination_walk'],
                'total_walking': total_walking,
                'requires_taxi_suggestion': requires_taxi_suggestion,
                'taxi_suggestion': taxi_suggestion,
                'preference': 'graph_based',
                'sort_priority': {
                    'transfers': route_result['transfers'],
                    'total_time': total_time,
                    'total_walking': total_walking
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing graph route result: {e}")
            return {}
    
    def _generate_taxi_suggestion(self, total_walking):
        """
        Generate taxi suggestion message based on walking distance
        
        Args:
            total_walking: Total walking distance in meters
            
        Returns:
            Taxi suggestion message
        """
        if total_walking <= 500:
            return None
        
        if total_walking <= 1000:
            return {
                'type': 'warning',
                'title': 'Long Walking Distance',
                'message': f'This route requires {total_walking:.0f}m of walking. Consider taking a taxi for part of the journey.',
                'walking_distance': total_walking,
                'suggestion': 'Consider taxi for convenience'
            }
        else:
            return {
                'type': 'recommendation',
                'title': 'Taxi Recommended',
                'message': f'This route requires {total_walking:.0f}m of walking. We recommend taking a taxi for a more comfortable journey.',
                'walking_distance': total_walking,
                'suggestion': 'Taxi recommended for comfort'
            }
    
    def _calculate_route_time(self, route_result, legs):
        """
        Calculate estimated total time for the route
        
        Args:
            route_result: Route result from graph BFS
            legs: Route legs
            
        Returns:
            Estimated total time in minutes
        """
        # Walking time (5 km/h)
        walking_speed_kmh = 5.0
        walking_time = (route_result['origin_walk'] + route_result['destination_walk']) / 1000 / walking_speed_kmh * 60
        
        # Bus time (20 km/h average)
        bus_speed_kmh = 20.0
        bus_distance = route_result['total_distance'] - route_result['origin_walk'] - route_result['destination_walk']
        bus_time = bus_distance / 1000 / bus_speed_kmh * 60
        
        # Transfer time (5 minutes per transfer)
        transfer_time = route_result['transfers'] * 5
        
        return walking_time + bus_time + transfer_time
    
    def _fallback_to_existing_bfs(self, origin, destination, routes, max_routes):
        """
        Fallback to existing BFS algorithm if graph-based approach fails
        
        Args:
            origin: Origin coordinates
            destination: Destination coordinates
            routes: Route documents
            max_routes: Maximum routes to return
            
        Returns:
            List of routes using existing BFS
        """
        try:
            from .graph_routing import bfs_multi_leg
            
            # Convert routes to expected format
            lines = []
            for route in routes:
                lines.append({
                    'line_id': route['line_id'],
                    'name': route['name'],
                    'type': route.get('type', 'bus'),
                    'direction': route.get('direction', 'unknown'),
                    'route': route['route']
                })
            
            # Use existing BFS
            results = bfs_multi_leg(
                lines, origin, destination,
                entry_thresh=300, exit_thresh=300, transfer_thresh=400,
                max_legs=4, min_bus_distance=300
            )
            
            # Process results
            processed_results = []
            for result in results[:max_routes]:
                processed_result = self._process_route_result(result, origin, destination)
                processed_results.append(processed_result)
            
            return processed_results
                
                except Exception as e:
            logger.error(f"Error in fallback BFS: {e}")
            return []
            
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
    
    def get_routes_near_point(self, point, max_distance_m=5000, projection=None):
        """
        Get routes near a specific point using MongoDB's $near operator.
        This is more efficient than $geoWithin for point-based queries.
        
        Args:
            point: [lng, lat] coordinates
            max_distance_m: Maximum distance in meters
            projection: MongoDB projection dict
            
        Returns:
            List of route documents sorted by distance
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
        
        query = {
            'route.coordinates': {
                '$near': {
                    '$geometry': {
                        'type': 'Point',
                        'coordinates': point
                    },
                    '$maxDistance': max_distance_m
                }
            }
        }
        
        return list(self.collection.find(query, projection).limit(self.max_routes_per_query))
    
    def get_routes_with_entry_exit_points(self, origin, destination, max_distance_m=5000):
        """
        Get routes with pre-calculated entry/exit points for efficient routing.
        
        Args:
            origin: [lng, lat] of origin
            destination: [lng, lat] of destination
            max_distance_m: Maximum distance to consider
            
        Returns:
            List of routes with entry/exit points
        """
        # Get routes near origin and destination
        origin_routes = self.get_routes_near_point(origin, max_distance_m)
        dest_routes = self.get_routes_near_point(destination, max_distance_m)
        
        # Combine and deduplicate routes
        all_routes = {}
        for route in origin_routes + dest_routes:
            line_id = route['line_id']
            if line_id not in all_routes:
                all_routes[line_id] = route
                all_routes[line_id]['entry_points'] = []
                all_routes[line_id]['exit_points'] = []
        
        # Calculate entry/exit points for each route
        for line_id, route in all_routes.items():
            coords = route['route']['coordinates']
            
            # Find entry points (near origin)
            entry_point, entry_dist, entry_index = nearest_point_on_line(origin, coords)
            if entry_dist <= max_distance_m:
                route['entry_points'].append({
                    'point': entry_point,
                    'distance': entry_dist,
                    'index': entry_index
                })
            
            # Find exit points (near destination)
            exit_point, exit_dist, exit_index = nearest_point_on_line(destination, coords)
            if exit_dist <= max_distance_m:
                route['exit_points'].append({
                    'point': exit_point,
                    'distance': exit_dist,
                    'index': exit_index
                })
        
        # Filter routes that have both entry and exit points
        valid_routes = [
            route for route in all_routes.values()
            if route['entry_points'] and route['exit_points']
        ]
        
        return valid_routes
    
    def optimize_coordinates_for_routing(self, coordinates):
        """
        Optimize coordinates for routing by applying RDP simplification if enabled.
        
        Args:
            coordinates: List of [lng, lat] coordinates
            
        Returns:
            Optimized coordinate list
        """
        if not self.simplification_enabled or len(coordinates) <= self.target_coordinates_per_route:
            return coordinates
        
        # Apply RDP simplification
        simplified = CoordinateSimplifier.rdp_simplification(
            coordinates, self.simplification_tolerance_m
        )
        
        # If still too many points, use adaptive simplification
        if len(simplified) > self.target_coordinates_per_route:
            simplified = CoordinateSimplifier.adaptive_simplification(
                simplified, self.target_coordinates_per_route
            )
        
        logger.debug(f"Simplified coordinates from {len(coordinates)} to {len(simplified)} points")
        return simplified
    
    def create_route_with_dual_coordinates(self, route_data):
        """
        Create route document with both simplified and full coordinates
        
        Args:
            route_data: Original route data with full coordinates
            
        Returns:
            Route data with both simplified and full coordinates
        """
        if 'route' not in route_data or 'coordinates' not in route_data['route']:
            return route_data
        
        full_coords = route_data['route']['coordinates']
        
        # Create simplified coordinates for routing
        simplified_coords = self.optimize_coordinates_for_routing(full_coords)
        
        # Store both versions
        route_data['route']['coordinates'] = full_coords  # Keep full coordinates
        route_data['route']['simplified_coordinates'] = simplified_coords  # Add simplified version
        
        # Add metadata about simplification
        route_data['route']['simplification_metadata'] = {
            'original_points': len(full_coords),
            'simplified_points': len(simplified_coords),
            'reduction_percentage': round((1 - len(simplified_coords) / len(full_coords)) * 100, 2),
            'tolerance_meters': self.simplification_tolerance_m
        }
        
        return route_data
    
    def get_coordinates_for_use_case(self, route_data, use_case='routing'):
        """
        Get appropriate coordinates based on use case
        
        Args:
            route_data: Route data with both coordinate versions
            use_case: 'routing' for simplified, 'rendering' for full coordinates
            
        Returns:
            Coordinate list for the specified use case
        """
        if 'route' not in route_data:
            return []
        
        if use_case == 'routing' and 'simplified_coordinates' in route_data['route']:
            return route_data['route']['simplified_coordinates']
        elif use_case == 'rendering' and 'coordinates' in route_data['route']:
            return route_data['route']['coordinates']
        else:
            # Fallback to available coordinates
            return route_data['route'].get('coordinates', [])
    
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
        Uses advanced geospatial queries with 500m bounding box buffer.
        
        Args:
            origin: [lng, lat] of origin
            destination: [lng, lat] of destination
            max_distance_km: Maximum distance to consider for route search
            
        Returns:
            List of relevant route documents with optimized coordinates
        """
        # Use bounding box filtering with 500m buffer
        routes = self.get_routes_with_bounding_box_filter(origin, destination, buffer_m=500)
        
        # Apply coordinate optimization to each route
        for route in routes:
            if 'route' in route and 'coordinates' in route['route']:
                original_coords = route['route']['coordinates']
                optimized_coords = self.optimize_coordinates_for_routing(original_coords)
                route['route']['coordinates'] = optimized_coords
                
                # Update entry/exit point indices if coordinates were simplified
                if len(optimized_coords) != len(original_coords):
                    self._update_entry_exit_indices(route, original_coords, optimized_coords)
        
        return routes
    
    def get_routes_with_bounding_box_filter(self, origin, destination, buffer_m=500):
        """
        Get routes using bounding box filtering with specified buffer
        
        Args:
            origin: [lng, lat] of origin
            destination: [lng, lat] of destination
            buffer_m: Buffer distance in meters around the bounding box
            
        Returns:
            List of routes within the buffered bounding box
        """
        # Calculate bounding box around origin and destination
        buffer_degrees = buffer_m / 111000.0  # Approximate meters to degrees
        
        min_lng = min(origin[0], destination[0]) - buffer_degrees
        max_lng = max(origin[0], destination[0]) + buffer_degrees
        min_lat = min(origin[1], destination[1]) - buffer_degrees
        max_lat = max(origin[1], destination[1]) + buffer_degrees
        
        # Use $geoIntersects for efficient bounding box queries
        query = {
            'route.coordinates': {
                '$geoIntersects': {
                    '$geometry': {
                        'type': 'Polygon',
                        'coordinates': [[
                            [min_lng, min_lat],
                            [max_lng, min_lat],
                            [max_lng, max_lat],
                            [min_lng, max_lat],
                            [min_lng, min_lat]  # Close the polygon
                        ]]
                    }
                }
            }
        }
        
        # Projection for routing - only essential fields
        projection = {
            'line_id': 1,
            'name': 1,
            'type': 1,
            'direction': 1,
            'route.coordinates': 1,
            '_id': 0
        }
        
        # Execute query with timeout and limit
        routes = list(self.collection.find(query, projection)
                     .limit(self.max_routes_per_query)
                     .max_time_ms(5000))  # 5 second timeout
        
        logger.info(f"Found {len(routes)} routes within bounding box (buffer: {buffer_m}m)")
        return routes
    
    def _update_entry_exit_indices(self, route, original_coords, optimized_coords):
        """
        Update entry/exit point indices after coordinate simplification.
        
        Args:
            route: Route document
            original_coords: Original coordinate list
            optimized_coords: Simplified coordinate list
        """
        # Find closest points in optimized coordinates for each entry/exit point
        for entry_point in route.get('entry_points', []):
            if 'index' in entry_point:
                original_index = entry_point['index']
                if original_index < len(original_coords):
                    original_point = original_coords[original_index]
                    # Find closest point in optimized coordinates
                    closest_index = 0
                    min_distance = float('inf')
                    for i, opt_point in enumerate(optimized_coords):
                        dist = distance(original_point, opt_point)
                        if dist < min_distance:
                            min_distance = dist
                            closest_index = i
                    entry_point['index'] = closest_index
        
        for exit_point in route.get('exit_points', []):
            if 'index' in exit_point:
                original_index = exit_point['index']
                if original_index < len(original_coords):
                    original_point = original_coords[original_index]
                    # Find closest point in optimized coordinates
                    closest_index = 0
                    min_distance = float('inf')
                    for i, opt_point in enumerate(optimized_coords):
                        dist = distance(original_point, opt_point)
                        if dist < min_distance:
                            min_distance = dist
                            closest_index = i
                    exit_point['index'] = closest_index
    
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
            # This is the most important index for performance
            self.collection.create_index([('route.coordinates', '2dsphere')])
            
            # Compound index for common queries
            self.collection.create_index([('type', 1), ('direction', 1)])
            
            # Compound geospatial index for type + coordinates
            self.collection.create_index([
                ('type', 1),
                ('route.coordinates', '2dsphere')
            ])
            
            # Text index for route names (if needed for search)
            self.collection.create_index([('name', 'text')])
            
            # Index for direction-based queries
            self.collection.create_index('direction')
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")
    
    def create_optimized_indexes(self):
        """
        Create advanced indexes for high-performance geospatial queries.
        Includes partial indexes and sparse indexes for better performance.
        """
        try:
            # Basic indexes
            self.create_indexes()
            
            # Partial index for active routes only
            self.collection.create_index(
                [('route.coordinates', '2dsphere')],
                partialFilterExpression={'type': {'$in': ['bus', 'minibus']}}
            )
            
            # Sparse index for routes with direction
            self.collection.create_index(
                'direction',
                sparse=True
            )
            
            # Compound sparse index
            self.collection.create_index(
                [('type', 1), ('direction', 1)],
                sparse=True
            )
            
            logger.info("Optimized MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating optimized MongoDB indexes: {e}")
    
    def analyze_query_performance(self, query_type='geospatial'):
        """
        Analyze query performance and provide optimization recommendations.
        
        Args:
            query_type: Type of query to analyze ('geospatial', 'text', 'compound')
            
        Returns:
            Dict with performance analysis and recommendations
        """
        try:
            analysis = {
                'query_type': query_type,
                'indexes_used': [],
                'recommendations': [],
                'performance_metrics': {}
            }
            
            # Get index usage statistics
            index_stats = self.collection.index_information()
            analysis['total_indexes'] = len(index_stats)
            
            # Check for geospatial index
            has_geospatial = any(
                '2dsphere' in str(index) for index in index_stats.values()
            )
            
            if not has_geospatial:
                analysis['recommendations'].append(
                    "Missing 2dsphere index on route.coordinates - critical for geospatial queries"
                )
            
            # Check for compound indexes
            has_compound = any(
                len(index) > 1 for index in index_stats.values()
            )
            
            if not has_compound:
                analysis['recommendations'].append(
                    "Consider adding compound indexes for common query patterns"
                )
            
            # Get collection statistics
            stats = self.db.command('collStats', self.collection.name)
            analysis['performance_metrics'] = {
                'total_documents': stats.get('count', 0),
                'avg_document_size': stats.get('avgObjSize', 0),
                'total_size': stats.get('size', 0),
                'index_size': stats.get('totalIndexSize', 0)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query performance: {e}")
            return {'error': str(e)}
    
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