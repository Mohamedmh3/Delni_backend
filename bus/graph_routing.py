from geopy.distance import geodesic
from pymongo import MongoClient
from collections import deque, defaultdict
import logging

# Set up logging
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def distance(p1, p2):
    return geodesic((p1[1], p1[0]), (p2[1], p2[0])).meters

def nearest_point_on_line(point, line_coords):
    min_dist = float('inf')
    nearest = None
    nearest_index = -1
    for i, coord in enumerate(line_coords):
        d = distance(point, coord)
        if d < min_dist:
            min_dist = d
            nearest = coord
            nearest_index = i
    return nearest, min_dist, nearest_index

def find_transfer_points(coordsA, coordsB, threshold=400):
    for a in coordsA:
        for b in coordsB:
            d = distance(a, b)
            if d < threshold:
                return a, b, d
    return None, None, float('inf')

def calculate_forward_bus_distance(coords, entry_index, exit_index):
    """
    Calculate bus distance from entry_index to exit_index, only going forward.
    If exit_index is before entry_index, return 0 (invalid route).
    """
    if exit_index <= entry_index:
        return 0  # Invalid - can't go backwards
    
    total_distance = 0
    for i in range(entry_index, exit_index):
        total_distance += distance(coords[i], coords[i + 1])
    return total_distance

def is_valid_route_segment(coords, entry_index, exit_index):
    """
    Check if the route segment from entry_index to exit_index is valid
    (entry comes before exit in the coordinate sequence).
    """
    return exit_index > entry_index

# --- Graph Construction ---
def build_transfer_graph(lines, transfer_threshold=400):
    # Build adjacency list: line_id -> list of (neighbor_line_id, transfer_point_A, transfer_point_B, transfer_dist)
    graph = defaultdict(list)
    for i, lineA in enumerate(lines):
        coordsA = lineA['route']['coordinates']
        directionA = lineA.get('direction', 'unknown')
        for j, lineB in enumerate(lines):
            if i == j:
                continue
            coordsB = lineB['route']['coordinates']
            directionB = lineB.get('direction', 'unknown')
            # Only allow transfers between routes with compatible directions
            if directionA == directionB:
                a, b, d = find_transfer_points(coordsA, coordsB, threshold=transfer_threshold)
                if d < transfer_threshold:
                    graph[lineA['line_id']].append({
                        'neighbor': lineB['line_id'],
                        'transfer_A': a,
                        'transfer_B': b,
                        'transfer_dist': d
                    })
            else:
                # For different directions, only allow transfer if logical (e.g., at endpoints)
                a, b, d = find_transfer_points(coordsA, coordsB, threshold=transfer_threshold)
                if d < transfer_threshold and is_logical_transfer(coordsA, coordsB, a, b, directionA, directionB):
                    graph[lineA['line_id']].append({
                        'neighbor': lineB['line_id'],
                        'transfer_A': a,
                        'transfer_B': b,
                        'transfer_dist': d
                    })
    return graph

def is_logical_transfer(coordsA, coordsB, pointA, pointB, directionA, directionB):
    if not pointA or not pointB:
        return False
    startA = coordsA[0]
    endA = coordsA[-1]
    startB = coordsB[0]
    endB = coordsB[-1]
    distA_to_start = distance(pointA, startA)
    distA_to_end = distance(pointA, endA)
    distB_to_start = distance(pointB, startB)
    distB_to_end = distance(pointB, endB)
    # Only allow transfer if at endpoints (e.g., end of one to start of another)
    if (distA_to_end < 200 and distB_to_start < 200) or (distA_to_start < 200 and distB_to_end < 200):
        return True
    return False

# --- BFS Pathfinding ---
def bfs_multi_leg(lines, origin, dest, entry_thresh=10000, exit_thresh=10000, transfer_thresh=400, max_legs=4, min_bus_distance=300):
    # Map line_id to line object
    line_map = {line['line_id']: line for line in lines}
    # Find all lines near origin
    origin_lines = []
    for line in lines:
        coords = line['route']['coordinates']
        entry, d_entry, entry_index = nearest_point_on_line(origin, coords)
        if d_entry < entry_thresh:
            origin_lines.append((line['line_id'], entry, d_entry, entry_index))
    # Find all lines near destination
    dest_lines = set()
    for line in lines:
        coords = line['route']['coordinates']
        exit, d_exit, exit_index = nearest_point_on_line(dest, coords)
        if d_exit < exit_thresh:
            dest_lines.add(line['line_id'])
    # Build transfer graph
    graph = build_transfer_graph(lines, transfer_threshold=transfer_thresh)
    # BFS: (current_line, path_so_far, transfer_points, walking_so_far, entry_point, entry_walk, entry_index)
    queue = deque()
    for line_id, entry, entry_walk, entry_index in origin_lines:
        queue.append((line_id, [line_id], [entry], entry_walk, entry, entry_walk, entry_index))
    visited = set()
    results = []
    max_simple_routes = 3  # Stop after finding 3 simple routes (1-2 buses)
    
    while queue:
        curr, path, transfers, walk, last_point, last_walk, last_entry_index = queue.popleft()
        if (curr, tuple(path)) in visited or len(path) > max_legs:
            continue
        visited.add((curr, tuple(path)))
        # Check if current line is near destination
        coords = line_map[curr]['route']['coordinates']
        exit, d_exit, exit_index = nearest_point_on_line(dest, coords)
        if curr in dest_lines and d_exit < exit_thresh:
            if is_valid_route_segment(coords, last_entry_index, exit_index):
                bus_distance_leg = calculate_forward_bus_distance(coords, last_entry_index, exit_index)
                logger.debug(f"Final leg check: line={curr}, bus_distance={bus_distance_leg:.2f}m")
                
                # 🚫 Skip if final leg bus distance is too short
                if bus_distance_leg < min_bus_distance:
                    logger.debug(f"Skipped final leg due to short bus distance: {bus_distance_leg:.2f}m < {min_bus_distance}m")
                    continue       
                
                logger.debug(f"Direct route found: line={curr}, bus_distance={bus_distance_leg:.2f}m, min_threshold={min_bus_distance}")
                
                # Check if bus distance meets minimum requirement
                if bus_distance_leg >= min_bus_distance:
                    if is_logical_route_sequence(path, line_map):
                        route_result = {
                            'lines': path,
                            'transfers': transfers,
                            'total_walking': walk + d_exit,
                            'legs': path,
                            'entry_points': transfers,
                            'exit_point': exit,
                            'exit_walk': d_exit
                        }
                        
                        # Add filtered polyline to the route result
                        filtered_polyline = extract_filtered_polyline(route_result, line_map)
                        route_result['filtered_polyline'] = filtered_polyline
                        
                        results.append(route_result)
                        logger.debug(f"Added direct route: {path}")
                        
                        # Early termination: if we have enough simple routes, stop searching
                        simple_routes = [r for r in results if len(r['lines']) <= 2]
                        if len(simple_routes) >= max_simple_routes:
                            logger.info(f"Found {len(simple_routes)} simple routes, stopping search")
                            break
                else:
                    logger.debug(f"Skipped direct route due to short bus distance: {bus_distance_leg:.2f}m < {min_bus_distance}m")
            continue
        # Explore neighbors
        for neighbor in graph[curr]:
            n_id = neighbor['neighbor']
            if n_id in path:
                continue  # avoid cycles
            
            # Find the CLOSEST intersection between the two routes after the current board point
            curr_coords = line_map[curr]['route']['coordinates']
            next_coords = line_map[n_id]['route']['coordinates']
            
            # Find the closest transfer point that is AFTER the current board point
            closest_transfer_dist = float('inf')
            best_transfer_A = None
            best_transfer_B = None
            best_transfer_A_index = -1
            best_transfer_B_index = -1
            
            # Search for the closest intersection after the current board point
            for i, point_A in enumerate(curr_coords):
                # Must be after the current board point (forward travel only)
                if i <= last_entry_index:
                    continue
                
                # Find closest point on the next route to this point
                for j, point_B in enumerate(next_coords):
                    dist = distance(point_A, point_B)
                    if dist < closest_transfer_dist and dist < 100:  # Within 100m
                        closest_transfer_dist = dist
                        best_transfer_A = point_A
                        best_transfer_B = point_B
                        best_transfer_A_index = i
                        best_transfer_B_index = j
            
            # If we found a valid transfer point, check bus distance before adding transfer
            if best_transfer_A and best_transfer_B:
                # Calculate bus distance from current entry point to transfer point
                bus_distance_to_transfer = calculate_forward_bus_distance(
                    curr_coords, last_entry_index, best_transfer_A_index
                )
                
                logger.debug(f"Transfer found: {curr} -> {n_id}, bus_distance={bus_distance_to_transfer:.2f}m, min_threshold={min_bus_distance}")
                
                # Skip transfer if bus distance is less than minimum threshold
                if bus_distance_to_transfer < min_bus_distance:
                    logger.debug(f"Skipped transfer due to short bus distance: {bus_distance_to_transfer:.2f}m < {min_bus_distance}m")
                    continue  # Don't add this transfer - bus distance too short
                
                # Calculate actual transfer walking distance
                actual_transfer_walk = distance(last_point, best_transfer_A)

                # Skip if walking distance between transfer points is below threshold
                if actual_transfer_walk < transfer_thresh:
                    logger.debug(f"Skipped transfer due to short walking transfer: {actual_transfer_walk:.2f}m < {transfer_thresh}m")
                    continue

                queue.append((n_id, path + [n_id], transfers + [best_transfer_A], walk + actual_transfer_walk, best_transfer_A, actual_transfer_walk, best_transfer_B_index))
                logger.debug(f"Added transfer: {curr} -> {n_id}")
    # Sort by simplicity first (fewer transfers), then by walking distance
    results.sort(key=lambda x: (len(x['lines']), x['total_walking']))
    
    # Limit results to only the best simple routes (max 5 results)
    max_results = 5
    return results[:max_results]

def is_logical_route_sequence(route_sequence, line_map):
    if len(route_sequence) <= 1:
        return True
    directions = [line_map[route_id].get('direction', 'unknown') for route_id in route_sequence]
    if len(set(directions)) == 1:
        return True
    for i in range(len(route_sequence) - 1):
        route1 = line_map[route_sequence[i]]
        route2 = line_map[route_sequence[i + 1]]
        if route1.get('direction') != route2.get('direction'):
            if not are_complementary_routes(route1, route2):
                return False
    return True

def extract_filtered_polyline(route, line_map):
    """
    Extract filtered polyline representing only the actual riding segments.
    
    Args:
        route: Route result from bfs_multi_leg with 'lines', 'entry_points', 'exit_point'
        line_map: Dictionary mapping line_id to line object
    
    Returns:
        List of coordinates representing only the bus riding segments
    """
    filtered_coords = []
    
    for i, line_id in enumerate(route['lines']):
        line_coords = line_map[line_id]['route']['coordinates']
        
        if i == 0:
            # First leg — from entry point to transfer or final stop
            entry_point = route['entry_points'][i]
            entry_index = find_closest_coordinate_index(line_coords, entry_point)
            
            if len(route['lines']) == 1:
                # Single leg route — get exit point too
                exit_point = route['exit_point']
                exit_index = find_closest_coordinate_index(line_coords, exit_point)
                if entry_index != -1 and exit_index != -1:
                    # Ensure proper order
                    start_idx = min(entry_index, exit_index)
                    end_idx = max(entry_index, exit_index)
                    filtered_coords.extend(line_coords[start_idx:end_idx + 1])
            else:
                # First leg of multi-leg route — go to transfer point
                transfer_point = route['entry_points'][i + 1]
                transfer_index = find_closest_coordinate_index(line_coords, transfer_point)
                if entry_index != -1 and transfer_index != -1:
                    # Ensure proper order
                    start_idx = min(entry_index, transfer_index)
                    end_idx = max(entry_index, transfer_index)
                    filtered_coords.extend(line_coords[start_idx:end_idx + 1])
                    
        elif i == len(route['lines']) - 1:
            # Final leg — from transfer to exit point
            entry_point = route['entry_points'][i]
            exit_point = route['exit_point']
            entry_index = find_closest_coordinate_index(line_coords, entry_point)
            exit_index = find_closest_coordinate_index(line_coords, exit_point)
            if entry_index != -1 and exit_index != -1:
                # Ensure proper order
                start_idx = min(entry_index, exit_index)
                end_idx = max(entry_index, exit_index)
                filtered_coords.extend(line_coords[start_idx:end_idx + 1])
                
        else:
            # Middle leg — from transfer to transfer
            entry_point = route['entry_points'][i]
            next_transfer_point = route['entry_points'][i + 1]
            entry_index = find_closest_coordinate_index(line_coords, entry_point)
            transfer_index = find_closest_coordinate_index(line_coords, next_transfer_point)
            if entry_index != -1 and transfer_index != -1:
                # Ensure proper order
                start_idx = min(entry_index, transfer_index)
                end_idx = max(entry_index, transfer_index)
                filtered_coords.extend(line_coords[start_idx:end_idx + 1])
    
    return filtered_coords

def find_closest_coordinate_index(coordinates, target_point):
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

def are_complementary_routes(route1, route2):
    coords1 = route1['route']['coordinates']
    coords2 = route2['route']['coordinates']
    start1, end1 = coords1[0], coords1[-1]
    start2, end2 = coords2[0], coords2[-1]
    if distance(end1, start2) < 500 or distance(end2, start1) < 500:
        return True
    for coord1 in coords1:
        for coord2 in coords2:
            if distance(coord1, coord2) < 200:
                return True
    return False 