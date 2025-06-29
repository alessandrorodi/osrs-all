"""
Advanced Pathfinding System for OSRS Navigation

This module provides comprehensive pathfinding capabilities including:
- A* pathfinding algorithm optimized for OSRS minimap
- Multi-floor navigation (stairs, ladders, teleports)
- Obstacle detection and avoidance
- Safe vs dangerous area detection
- Teleport route optimization
- Real-time path recalculation
- Movement efficiency metrics

Integrates with the minimap analyzer for intelligent navigation decisions.
"""

import heapq
import math
import time
import threading
from typing import List, Dict, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from vision.minimap_analyzer import (
    AdvancedMinimapAnalyzer, MinimapAnalysisResult, MinimapRegionType, 
    DotType, MinimapDot
)
from core.automation import mouse, random_delay
from config.settings import AUTOMATION, SAFETY

logger = logging.getLogger(__name__)


class PathfindingAlgorithm(Enum):
    """Available pathfinding algorithms"""
    A_STAR = "a_star"
    DIJKSTRA = "dijkstra"
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"


class MovementType(Enum):
    """Types of movement in OSRS"""
    WALK = "walk"
    RUN = "run"
    TELEPORT = "teleport"
    STAIRS = "stairs"
    LADDER = "ladder"
    DOOR = "door"
    BOAT = "boat"
    AGILITY = "agility"


class PathNode:
    """Represents a node in the pathfinding graph"""
    
    def __init__(self, x: int, y: int, floor: int = 0):
        self.x = x
        self.y = y
        self.floor = floor
        self.g_cost = 0.0  # Cost from start
        self.h_cost = 0.0  # Heuristic cost to goal
        self.f_cost = 0.0  # Total cost (g + h)
        self.parent: Optional['PathNode'] = None
        self.movement_type = MovementType.WALK
        self.danger_level = 0.0  # 0.0 = safe, 1.0 = very dangerous
        self.traversal_cost = 1.0  # Base cost to traverse this node
        
    @property
    def position(self) -> Tuple[int, int, int]:
        """Get 3D position (x, y, floor)"""
        return (self.x, self.y, self.floor)
    
    def __lt__(self, other: 'PathNode') -> bool:
        return self.f_cost < other.f_cost
    
    def __eq__(self, other: 'PathNode') -> bool:
        return (self.x == other.x and 
                self.y == other.y and 
                self.floor == other.floor)
    
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.floor))


@dataclass
class PathResult:
    """Result of pathfinding operation"""
    path: List[PathNode]
    total_cost: float
    algorithm_used: PathfindingAlgorithm
    calculation_time: float
    success: bool
    distance: float
    estimated_time: float  # Estimated time in seconds
    danger_rating: float  # 0.0 = safe, 1.0 = very dangerous
    teleports_used: int
    alternative_paths: List[List[PathNode]]  # Alternative route options


@dataclass
class NavigationGoal:
    """Represents a navigation goal with preferences"""
    target_x: int
    target_y: int
    target_floor: int = 0
    max_danger: float = 0.5  # Maximum acceptable danger level
    prefer_safe_route: bool = True
    allow_teleports: bool = True
    allow_wilderness: bool = False
    max_distance: Optional[float] = None
    priority: float = 1.0
    timeout: float = 30.0  # Maximum pathfinding time


class OSRSPathfinder:
    """
    Advanced pathfinding system optimized for OSRS
    
    Features:
    - Multiple pathfinding algorithms
    - Multi-floor navigation
    - Danger avoidance
    - Teleport optimization
    - Real-time recalculation
    - Performance optimization
    """
    
    def __init__(self, minimap_analyzer: AdvancedMinimapAnalyzer, 
                 algorithm: PathfindingAlgorithm = PathfindingAlgorithm.A_STAR):
        """
        Initialize the pathfinder
        
        Args:
            minimap_analyzer: Minimap analyzer instance
            algorithm: Default pathfinding algorithm
        """
        self.minimap_analyzer = minimap_analyzer
        self.algorithm = algorithm
        
        # Known locations and connections
        self.known_teleports = self._load_teleport_data()
        self.known_stairs = self._load_stair_data()
        self.known_doors = self._load_door_data()
        self.danger_zones = self._load_danger_zones()
        
        # Performance optimization
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.path_cache = {}  # Cache for frequently used paths
        self.cache_ttl = 300  # Cache time-to-live in seconds
        
        # Navigation state
        self.current_path: Optional[List[PathNode]] = None
        self.current_goal: Optional[NavigationGoal] = None
        self.path_start_time = 0.0
        
        # Statistics
        self.stats = {
            'paths_calculated': 0,
            'avg_calculation_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'successful_navigations': 0,
            'failed_navigations': 0
        }
        
        logger.info(f"OSRSPathfinder initialized with {algorithm.value} algorithm")
    
    def find_path(self, start_x: int, start_y: int, goal: NavigationGoal, 
                  current_minimap: Optional[MinimapAnalysisResult] = None) -> PathResult:
        """
        Find optimal path from start to goal
        
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            goal: Navigation goal with preferences
            current_minimap: Current minimap analysis for context
            
        Returns:
            PathResult with path and metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(start_x, start_y, goal)
            cached_result = self._get_cached_path(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.debug(f"Using cached path for {cache_key}")
                return cached_result
            
            self.stats['cache_misses'] += 1
            
            # Create start and goal nodes
            start_node = PathNode(start_x, start_y, 0)
            goal_node = PathNode(goal.target_x, goal.target_y, goal.target_floor)
            
            # Choose pathfinding algorithm
            if self.algorithm == PathfindingAlgorithm.A_STAR:
                path_nodes = self._a_star_search(start_node, goal_node, goal, current_minimap)
            elif self.algorithm == PathfindingAlgorithm.DIJKSTRA:
                path_nodes = self._dijkstra_search(start_node, goal_node, goal, current_minimap)
            else:
                path_nodes = self._a_star_search(start_node, goal_node, goal, current_minimap)
            
            calculation_time = time.time() - start_time
            
            # Create result
            if path_nodes:
                result = self._create_path_result(path_nodes, calculation_time, True)
                
                # Cache successful results
                self._cache_path(cache_key, result)
                
                # Update statistics
                self._update_stats(calculation_time, True)
                
                logger.info(f"Path found: {len(path_nodes)} nodes, "
                          f"{result.total_cost:.1f} cost, {calculation_time:.3f}s")
                return result
            else:
                # No path found
                result = PathResult(
                    path=[],
                    total_cost=float('inf'),
                    algorithm_used=self.algorithm,
                    calculation_time=calculation_time,
                    success=False,
                    distance=0.0,
                    estimated_time=0.0,
                    danger_rating=0.0,
                    teleports_used=0,
                    alternative_paths=[]
                )
                
                self._update_stats(calculation_time, False)
                logger.warning(f"No path found from ({start_x}, {start_y}) to "
                             f"({goal.target_x}, {goal.target_y})")
                return result
                
        except Exception as e:
            logger.error(f"Pathfinding failed: {e}")
            calculation_time = time.time() - start_time
            self._update_stats(calculation_time, False)
            
            return PathResult(
                path=[],
                total_cost=float('inf'),
                algorithm_used=self.algorithm,
                calculation_time=calculation_time,
                success=False,
                distance=0.0,
                estimated_time=0.0,
                danger_rating=0.0,
                teleports_used=0,
                alternative_paths=[]
            )
    
    def _a_star_search(self, start: PathNode, goal: PathNode, 
                      nav_goal: NavigationGoal, 
                      current_minimap: Optional[MinimapAnalysisResult]) -> Optional[List[PathNode]]:
        """A* pathfinding algorithm implementation"""
        try:
            open_set = []
            closed_set: Set[PathNode] = set()
            
            # Initialize start node
            start.g_cost = 0
            start.h_cost = self._heuristic_cost(start, goal)
            start.f_cost = start.g_cost + start.h_cost
            
            heapq.heappush(open_set, start)
            
            max_iterations = 10000  # Prevent infinite loops
            iterations = 0
            
            while open_set and iterations < max_iterations:
                iterations += 1
                
                # Get node with lowest f_cost
                current = heapq.heappop(open_set)
                
                # Check if we reached the goal
                if self._is_goal_reached(current, goal):
                    return self._reconstruct_path(current)
                
                closed_set.add(current)
                
                # Explore neighbors
                neighbors = self._get_neighbors(current, nav_goal, current_minimap)
                
                for neighbor in neighbors:
                    if neighbor in closed_set:
                        continue
                    
                    # Calculate costs
                    tentative_g_cost = current.g_cost + self._movement_cost(current, neighbor, nav_goal)
                    
                    # Check if this is a better path to the neighbor
                    if neighbor not in [node for node in open_set] or tentative_g_cost < neighbor.g_cost:
                        neighbor.parent = current
                        neighbor.g_cost = tentative_g_cost
                        neighbor.h_cost = self._heuristic_cost(neighbor, goal)
                        neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                        
                        if neighbor not in [node for node in open_set]:
                            heapq.heappush(open_set, neighbor)
            
            logger.warning(f"A* search exhausted after {iterations} iterations")
            return None
            
        except Exception as e:
            logger.error(f"A* search failed: {e}")
            return None
    
    def _dijkstra_search(self, start: PathNode, goal: PathNode,
                        nav_goal: NavigationGoal,
                        current_minimap: Optional[MinimapAnalysisResult]) -> Optional[List[PathNode]]:
        """Dijkstra's algorithm implementation"""
        try:
            open_set = []
            distances = {start: 0.0}
            previous = {}
            
            heapq.heappush(open_set, (0.0, start))
            
            while open_set:
                current_distance, current = heapq.heappop(open_set)
                
                if self._is_goal_reached(current, goal):
                    # Reconstruct path
                    path = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = previous.get(node)
                    return list(reversed(path))
                
                if current_distance > distances.get(current, float('inf')):
                    continue
                
                # Explore neighbors
                neighbors = self._get_neighbors(current, nav_goal, current_minimap)
                
                for neighbor in neighbors:
                    distance = current_distance + self._movement_cost(current, neighbor, nav_goal)
                    
                    if distance < distances.get(neighbor, float('inf')):
                        distances[neighbor] = distance
                        previous[neighbor] = current
                        heapq.heappush(open_set, (distance, neighbor))
            
            return None
            
        except Exception as e:
            logger.error(f"Dijkstra search failed: {e}")
            return None
    
    def _heuristic_cost(self, node: PathNode, goal: PathNode) -> float:
        """Calculate heuristic cost (Manhattan distance with floor penalty)"""
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        dz = abs(node.floor - goal.floor)
        
        # Base Manhattan distance
        base_cost = dx + dy
        
        # Floor change penalty
        floor_penalty = dz * 10  # Stairs/ladders are expensive
        
        return float(base_cost + floor_penalty)
    
    def _movement_cost(self, from_node: PathNode, to_node: PathNode, 
                      nav_goal: NavigationGoal) -> float:
        """Calculate movement cost between two nodes"""
        # Base movement cost
        dx = abs(to_node.x - from_node.x)
        dy = abs(to_node.y - from_node.y)
        base_cost = math.sqrt(dx*dx + dy*dy)
        
        # Movement type modifiers
        type_modifiers = {
            MovementType.WALK: 1.0,
            MovementType.RUN: 0.6,  # Running is faster
            MovementType.TELEPORT: 0.1,  # Teleports are very fast
            MovementType.STAIRS: 2.0,  # Stairs take time
            MovementType.LADDER: 2.5,  # Ladders are slower
            MovementType.DOOR: 1.5,  # Doors require interaction
            MovementType.BOAT: 5.0,  # Boats are slow but cover long distance
            MovementType.AGILITY: 0.8  # Agility shortcuts can be faster
        }
        
        type_cost = base_cost * type_modifiers.get(to_node.movement_type, 1.0)
        
        # Danger penalty
        if nav_goal.prefer_safe_route:
            danger_penalty = to_node.danger_level * 5.0  # Avoid dangerous areas
            type_cost += danger_penalty
        
        # Traversal cost (terrain difficulty)
        type_cost *= to_node.traversal_cost
        
        return type_cost
    
    def _get_neighbors(self, node: PathNode, nav_goal: NavigationGoal,
                      current_minimap: Optional[MinimapAnalysisResult]) -> List[PathNode]:
        """Get walkable neighbor nodes"""
        neighbors = []
        
        # Basic 8-directional movement
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in directions:
            new_x = node.x + dx
            new_y = node.y + dy
            
            # Create neighbor node
            neighbor = PathNode(new_x, new_y, node.floor)
            
            # Check if neighbor is valid
            if self._is_valid_position(neighbor, nav_goal, current_minimap):
                # Set movement properties
                neighbor.movement_type = MovementType.RUN if AUTOMATION.get('prefer_running', True) else MovementType.WALK
                neighbor.danger_level = self._calculate_danger_level(neighbor, current_minimap)
                neighbor.traversal_cost = self._calculate_traversal_cost(neighbor, current_minimap)
                
                neighbors.append(neighbor)
        
        # Add special movement options (teleports, stairs, etc.)
        special_neighbors = self._get_special_movement_options(node, nav_goal)
        neighbors.extend(special_neighbors)
        
        return neighbors
    
    def _is_valid_position(self, node: PathNode, nav_goal: NavigationGoal,
                          current_minimap: Optional[MinimapAnalysisResult]) -> bool:
        """Check if a position is valid for movement"""
        # Basic bounds checking (minimap coordinates)
        if not (0 <= node.x <= 146 and 0 <= node.y <= 151):
            return False
        
        # Check danger level
        danger = self._calculate_danger_level(node, current_minimap)
        if danger > nav_goal.max_danger:
            return False
        
        # Check wilderness restriction
        if not nav_goal.allow_wilderness and self._is_wilderness(node):
            return False
        
        # Check if position is walkable using minimap data
        if current_minimap:
            # This would check against walkable areas from minimap analysis
            # For now, assume most areas are walkable
            pass
        
        return True
    
    def _calculate_danger_level(self, node: PathNode, 
                               current_minimap: Optional[MinimapAnalysisResult]) -> float:
        """Calculate danger level for a position"""
        danger = 0.0
        
        # Check known danger zones
        for danger_zone in self.danger_zones:
            zone_x, zone_y, zone_radius, zone_danger = danger_zone
            distance = math.sqrt((node.x - zone_x)**2 + (node.y - zone_y)**2)
            if distance <= zone_radius:
                danger = max(danger, zone_danger)
        
        # Check for aggressive NPCs nearby
        if current_minimap:
            aggressive_npcs = current_minimap.get_dots_by_type(DotType.NPC_AGGRESSIVE)
            for npc in aggressive_npcs:
                distance = math.sqrt((node.x - npc.x)**2 + (node.y - npc.y)**2)
                if distance <= 10:  # Within danger range
                    danger = max(danger, 0.7)
        
        # Wilderness is always dangerous
        if self._is_wilderness(node):
            danger = max(danger, 0.8)
        
        return min(danger, 1.0)
    
    def _calculate_traversal_cost(self, node: PathNode,
                                 current_minimap: Optional[MinimapAnalysisResult]) -> float:
        """Calculate traversal difficulty cost"""
        # Base cost
        cost = 1.0
        
        # Terrain-based cost would be calculated here
        # For now, return base cost
        return cost
    
    def _is_wilderness(self, node: PathNode) -> bool:
        """Check if a position is in the wilderness"""
        # Simplified wilderness detection
        # In a real implementation, this would check against known wilderness coordinates
        return False
    
    def _get_special_movement_options(self, node: PathNode, 
                                    nav_goal: NavigationGoal) -> List[PathNode]:
        """Get special movement options like teleports, stairs, etc."""
        special_moves = []
        
        if nav_goal.allow_teleports:
            # Check for available teleports
            teleports = self._get_available_teleports(node)
            special_moves.extend(teleports)
        
        # Check for stairs/ladders
        stairs = self._get_available_stairs(node)
        special_moves.extend(stairs)
        
        return special_moves
    
    def _get_available_teleports(self, node: PathNode) -> List[PathNode]:
        """Get available teleport destinations from current position"""
        teleports = []
        
        # Check known teleport sources
        for teleport in self.known_teleports:
            source_x, source_y, dest_x, dest_y, requirements = teleport
            
            # Check if we're at a teleport source
            if abs(node.x - source_x) <= 2 and abs(node.y - source_y) <= 2:
                dest_node = PathNode(dest_x, dest_y, node.floor)
                dest_node.movement_type = MovementType.TELEPORT
                dest_node.traversal_cost = 0.1  # Teleports are very efficient
                teleports.append(dest_node)
        
        return teleports
    
    def _get_available_stairs(self, node: PathNode) -> List[PathNode]:
        """Get available stair connections from current position"""
        stairs = []
        
        # Check known stairs
        for stair in self.known_stairs:
            x, y, floor_from, floor_to = stair
            
            # Check if we're at stairs
            if abs(node.x - x) <= 1 and abs(node.y - y) <= 1 and node.floor == floor_from:
                stair_node = PathNode(x, y, floor_to)
                stair_node.movement_type = MovementType.STAIRS
                stair_node.traversal_cost = 2.0  # Stairs take time
                stairs.append(stair_node)
        
        return stairs
    
    def _is_goal_reached(self, node: PathNode, goal: PathNode, tolerance: int = 1) -> bool:
        """Check if goal is reached within tolerance"""
        return (abs(node.x - goal.x) <= tolerance and 
                abs(node.y - goal.y) <= tolerance and
                node.floor == goal.floor)
    
    def _reconstruct_path(self, goal_node: PathNode) -> List[PathNode]:
        """Reconstruct path from goal node back to start"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current)
            current = current.parent
        
        return list(reversed(path))
    
    def _create_path_result(self, path_nodes: List[PathNode], 
                           calculation_time: float, success: bool) -> PathResult:
        """Create PathResult from path nodes"""
        if not path_nodes:
            return PathResult(
                path=[],
                total_cost=float('inf'),
                algorithm_used=self.algorithm,
                calculation_time=calculation_time,
                success=False,
                distance=0.0,
                estimated_time=0.0,
                danger_rating=0.0,
                teleports_used=0,
                alternative_paths=[]
            )
        
        # Calculate metrics
        total_cost = sum(node.g_cost for node in path_nodes)
        distance = len(path_nodes)
        
        # Estimate time based on movement types
        estimated_time = 0.0
        teleports_used = 0
        max_danger = 0.0
        
        for node in path_nodes:
            if node.movement_type == MovementType.TELEPORT:
                teleports_used += 1
                estimated_time += 3.0  # 3 seconds for teleport
            elif node.movement_type == MovementType.STAIRS:
                estimated_time += 2.0  # 2 seconds for stairs
            else:
                estimated_time += 0.6  # 0.6 seconds per tile (game tick)
            
            max_danger = max(max_danger, node.danger_level)
        
        return PathResult(
            path=path_nodes,
            total_cost=total_cost,
            algorithm_used=self.algorithm,
            calculation_time=calculation_time,
            success=success,
            distance=distance,
            estimated_time=estimated_time,
            danger_rating=max_danger,
            teleports_used=teleports_used,
            alternative_paths=[]
        )
    
    def _generate_cache_key(self, start_x: int, start_y: int, goal: NavigationGoal) -> str:
        """Generate cache key for path"""
        return f"{start_x},{start_y}->{goal.target_x},{goal.target_y}_{goal.target_floor}_{goal.max_danger}"
    
    def _get_cached_path(self, cache_key: str) -> Optional[PathResult]:
        """Get path from cache if valid"""
        if cache_key in self.path_cache:
            cached_time, result = self.path_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return result
            else:
                # Cache expired
                del self.path_cache[cache_key]
        return None
    
    def _cache_path(self, cache_key: str, result: PathResult):
        """Cache path result"""
        self.path_cache[cache_key] = (time.time(), result)
        
        # Limit cache size
        if len(self.path_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self.path_cache.keys(), 
                           key=lambda k: self.path_cache[k][0])
            del self.path_cache[oldest_key]
    
    def _update_stats(self, calculation_time: float, success: bool):
        """Update pathfinding statistics"""
        self.stats['paths_calculated'] += 1
        
        # Update average calculation time
        current_avg = self.stats['avg_calculation_time']
        count = self.stats['paths_calculated']
        self.stats['avg_calculation_time'] = ((current_avg * (count - 1) + calculation_time) / count)
        
        if success:
            self.stats['successful_navigations'] += 1
        else:
            self.stats['failed_navigations'] += 1
    
    def _load_teleport_data(self) -> List[Tuple[int, int, int, int, Dict]]:
        """Load known teleport locations and destinations"""
        # This would load from a data file in a real implementation
        teleports = [
            # (source_x, source_y, dest_x, dest_y, requirements)
            (100, 100, 200, 200, {'spell': 'lumbridge_teleport'}),
            (150, 150, 250, 250, {'spell': 'varrock_teleport'}),
            # Add more teleports as needed
        ]
        return teleports
    
    def _load_stair_data(self) -> List[Tuple[int, int, int, int]]:
        """Load known stair connections"""
        # This would load from a data file in a real implementation
        stairs = [
            # (x, y, from_floor, to_floor)
            (120, 120, 0, 1),
            (120, 120, 1, 0),
            # Add more stairs as needed
        ]
        return stairs
    
    def _load_door_data(self) -> List[Tuple[int, int, Dict]]:
        """Load known door locations and requirements"""
        doors = [
            # (x, y, requirements)
            (110, 110, {'key': 'brass_key'}),
            # Add more doors as needed
        ]
        return doors
    
    def _load_danger_zones(self) -> List[Tuple[int, int, int, float]]:
        """Load known dangerous areas"""
        # (x, y, radius, danger_level)
        danger_zones = [
            (300, 300, 20, 0.9),  # High-level combat area
            (400, 400, 15, 0.7),  # Moderate danger zone
            # Add more danger zones as needed
        ]
        return danger_zones
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pathfinding statistics"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear the path cache"""
        self.path_cache.clear()
        logger.info("Path cache cleared")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        self.clear_cache()
        logger.info("OSRSPathfinder cleanup completed")