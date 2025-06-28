"""
Comprehensive Test Suite for Navigation System

This module provides comprehensive testing for:
- Advanced minimap analysis
- Pathfinding algorithms (A*, Dijkstra)
- Navigation GUI integration
- Performance benchmarks
- Real-time processing capabilities
- Error handling and edge cases

Tests cover all components of the minimap intelligence and navigation system.
"""

import unittest
import unittest.mock as mock
import numpy as np
import time
import threading
from typing import List, Optional, Tuple
from unittest.mock import patch, MagicMock
import logging

# Test imports
try:
    from vision.minimap_analyzer import (
        AdvancedMinimapAnalyzer, MinimapAnalysisResult, DotType, 
        MinimapRegionType, MinimapDot, CompassInfo, MinimapClickableArea
    )
    from navigation.pathfinding import (
        OSRSPathfinder, NavigationGoal, PathResult, PathNode, 
        MovementType, PathfindingAlgorithm
    )
    NAVIGATION_AVAILABLE = True
except ImportError as e:
    print(f"Navigation components not available: {e}")
    NAVIGATION_AVAILABLE = False

# Create mock components if not available
if not NAVIGATION_AVAILABLE:
    # Mock classes for testing when dependencies aren't available
    class AdvancedMinimapAnalyzer:
        pass
    class OSRSPathfinder:
        pass
    class MinimapAnalysisResult:
        pass
    class NavigationGoal:
        pass
    class PathResult:
        pass
    class PathNode:
        pass
    class DotType:
        pass
    class MovementType:
        pass


def create_test_minimap_image(width: int = 146, height: int = 151) -> np.ndarray:
    """Create a test minimap image for testing"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some test features
    # Walkable area (dark)
    image[50:100, 50:100] = [20, 30, 20]
    
    # Obstacles (white lines)
    image[40:41, 30:110] = [255, 255, 255]  # Horizontal wall
    image[30:110, 40:41] = [255, 255, 255]  # Vertical wall
    
    # Player position (center, white)
    image[73:76, 73:76] = [255, 255, 255]
    
    # NPCs (yellow dots)
    image[60:62, 80:82] = [0, 255, 255]
    image[90:92, 60:62] = [0, 255, 255]
    
    # Items (red dots)
    image[70:71, 90:91] = [0, 0, 255]
    
    return image


def create_test_game_screenshot(width: int = 765, height: int = 503) -> np.ndarray:
    """Create a test OSRS game screenshot"""
    screenshot = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add minimap region
    minimap = create_test_minimap_image()
    screenshot[9:160, 570:716] = minimap
    
    # Add compass
    screenshot[9:34, 570:595] = [100, 100, 100]  # Gray compass area
    
    return screenshot


@unittest.skipUnless(NAVIGATION_AVAILABLE, "Navigation system not available")
class TestAdvancedMinimapAnalyzer(unittest.TestCase):
    """Test cases for advanced minimap analyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = AdvancedMinimapAnalyzer(device="cpu", enable_yolo=False)
        self.test_image = create_test_minimap_image()
        self.test_screenshot = create_test_game_screenshot()
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'analyzer'):
            self.analyzer.cleanup()
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization"""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.device, "cpu")
        self.assertFalse(self.analyzer.enable_yolo)
        self.assertEqual(self.analyzer.minimap_region, (570, 9, 146, 151))
    
    def test_analyze_minimap_basic(self):
        """Test basic minimap analysis"""
        result = self.analyzer.analyze_minimap(self.test_screenshot)
        
        self.assertIsInstance(result, MinimapAnalysisResult)
        self.assertIsNotNone(result.timestamp)
        self.assertIsInstance(result.player_position, tuple)
        self.assertEqual(len(result.player_position), 2)
        self.assertIsInstance(result.detected_dots, list)
        self.assertIsInstance(result.processing_time, float)
        self.assertGreater(result.processing_time, 0)
    
    def test_player_position_detection(self):
        """Test player position detection"""
        result = self.analyzer.analyze_minimap(self.test_screenshot)
        
        # Player should be near center
        player_x, player_y = result.player_position
        self.assertGreater(player_x, 60)
        self.assertLess(player_x, 90)
        self.assertGreater(player_y, 60)
        self.assertLess(player_y, 90)
    
    def test_dot_detection(self):
        """Test dot detection on minimap"""
        result = self.analyzer.analyze_minimap(self.test_screenshot)
        
        # Should detect some dots
        self.assertGreater(len(result.detected_dots), 0)
        
        # Check dot properties
        for dot in result.detected_dots:
            self.assertIsInstance(dot, MinimapDot)
            self.assertIsInstance(dot.x, int)
            self.assertIsInstance(dot.y, int)
            self.assertIsInstance(dot.dot_type, DotType)
            self.assertGreaterEqual(dot.confidence, 0.0)
            self.assertLessEqual(dot.confidence, 1.0)
    
    def test_compass_analysis(self):
        """Test compass analysis"""
        result = self.analyzer.analyze_minimap(self.test_screenshot)
        
        compass_info = result.compass_info
        self.assertIsInstance(compass_info, CompassInfo)
        self.assertIsInstance(compass_info.north_angle, float)
        self.assertIsInstance(compass_info.camera_angle, float)
        self.assertIsInstance(compass_info.zoom_level, float)
        self.assertIsInstance(compass_info.is_compass_visible, bool)
    
    def test_region_classification(self):
        """Test region classification"""
        result = self.analyzer.analyze_minimap(self.test_screenshot)
        
        self.assertIsInstance(result.region_type, MinimapRegionType)
        self.assertIsInstance(result.region_name, str)
        self.assertGreater(len(result.region_name), 0)
    
    def test_clickable_areas_detection(self):
        """Test clickable areas detection"""
        result = self.analyzer.analyze_minimap(self.test_screenshot)
        
        self.assertIsInstance(result.clickable_areas, list)
        
        for area in result.clickable_areas:
            self.assertIsInstance(area, MinimapClickableArea)
            self.assertIsInstance(area.x, int)
            self.assertIsInstance(area.y, int)
            self.assertIsInstance(area.width, int)
            self.assertIsInstance(area.height, int)
            self.assertIsInstance(area.area_type, str)
            self.assertGreaterEqual(area.confidence, 0.0)
            self.assertLessEqual(area.confidence, 1.0)
    
    def test_obstacle_detection(self):
        """Test obstacle detection"""
        result = self.analyzer.analyze_minimap(self.test_screenshot)
        
        self.assertIsInstance(result.obstacles, list)
        # Should detect some obstacles from test image
        self.assertGreater(len(result.obstacles), 0)
        
        for obstacle in result.obstacles:
            self.assertIsInstance(obstacle, tuple)
            self.assertEqual(len(obstacle), 2)
            self.assertIsInstance(obstacle[0], int)
            self.assertIsInstance(obstacle[1], int)
    
    def test_walkable_areas_detection(self):
        """Test walkable areas detection"""
        result = self.analyzer.analyze_minimap(self.test_screenshot)
        
        self.assertIsInstance(result.walkable_areas, list)
        self.assertGreater(len(result.walkable_areas), 0)
        
        for area in result.walkable_areas:
            self.assertIsInstance(area, tuple)
            self.assertEqual(len(area), 2)
            self.assertIsInstance(area[0], int)
            self.assertIsInstance(area[1], int)
    
    def test_performance_stats(self):
        """Test performance statistics tracking"""
        # Run analysis multiple times
        for _ in range(5):
            self.analyzer.analyze_minimap(self.test_screenshot)
        
        stats = self.analyzer.get_performance_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('frames_processed', stats)
        self.assertIn('avg_processing_time', stats)
        self.assertIn('fps', stats)
        
        self.assertGreaterEqual(stats['frames_processed'], 5)
        self.assertGreater(stats['avg_processing_time'], 0)
    
    def test_is_point_walkable(self):
        """Test walkability checking"""
        # Test with walkable point
        walkable = self.analyzer.is_point_walkable(75, 75, self.test_image)
        self.assertIsInstance(walkable, bool)
        
        # Test with boundary conditions
        walkable_edge = self.analyzer.is_point_walkable(0, 0, self.test_image)
        self.assertIsInstance(walkable_edge, bool)
        
        # Test with out of bounds
        walkable_oob = self.analyzer.is_point_walkable(-1, -1, self.test_image)
        self.assertFalse(walkable_oob)
    
    def test_error_handling(self):
        """Test error handling in minimap analysis"""
        # Test with invalid image
        invalid_image = np.array([])
        result = self.analyzer.analyze_minimap(invalid_image)
        
        # Should return empty result without crashing
        self.assertIsInstance(result, MinimapAnalysisResult)
        self.assertEqual(result.player_position, (73, 75))  # Default position
        
        # Test with None image
        with self.assertLogs(level='ERROR'):
            result_none = self.analyzer.analyze_minimap(None)
            self.assertIsInstance(result_none, MinimapAnalysisResult)


@unittest.skipUnless(NAVIGATION_AVAILABLE, "Navigation system not available")
class TestOSRSPathfinder(unittest.TestCase):
    """Test cases for OSRS pathfinder"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = AdvancedMinimapAnalyzer(device="cpu", enable_yolo=False)
        self.pathfinder = OSRSPathfinder(self.analyzer)
        self.test_screenshot = create_test_game_screenshot()
        self.test_analysis = self.analyzer.analyze_minimap(self.test_screenshot)
    
    def tearDown(self):
        """Clean up after tests"""
        if hasattr(self, 'pathfinder'):
            self.pathfinder.cleanup()
        if hasattr(self, 'analyzer'):
            self.analyzer.cleanup()
    
    def test_pathfinder_initialization(self):
        """Test pathfinder initialization"""
        self.assertIsNotNone(self.pathfinder)
        self.assertEqual(self.pathfinder.algorithm, PathfindingAlgorithm.A_STAR)
        self.assertIsInstance(self.pathfinder.known_teleports, list)
        self.assertIsInstance(self.pathfinder.known_stairs, list)
        self.assertIsInstance(self.pathfinder.known_doors, list)
        self.assertIsInstance(self.pathfinder.danger_zones, list)
    
    def test_navigation_goal_creation(self):
        """Test navigation goal creation"""
        goal = NavigationGoal(
            target_x=100,
            target_y=100,
            max_danger=0.5,
            prefer_safe_route=True,
            allow_teleports=True,
            allow_wilderness=False
        )
        
        self.assertEqual(goal.target_x, 100)
        self.assertEqual(goal.target_y, 100)
        self.assertEqual(goal.max_danger, 0.5)
        self.assertTrue(goal.prefer_safe_route)
        self.assertTrue(goal.allow_teleports)
        self.assertFalse(goal.allow_wilderness)
    
    def test_path_finding_basic(self):
        """Test basic pathfinding"""
        goal = NavigationGoal(target_x=80, target_y=80)
        result = self.pathfinder.find_path(70, 70, goal, self.test_analysis)
        
        self.assertIsInstance(result, PathResult)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.path, list)
        self.assertIsInstance(result.total_cost, float)
        self.assertIsInstance(result.calculation_time, float)
        self.assertGreater(result.calculation_time, 0)
    
    def test_path_node_creation(self):
        """Test path node creation and properties"""
        node = PathNode(10, 20, 0)
        
        self.assertEqual(node.x, 10)
        self.assertEqual(node.y, 20)
        self.assertEqual(node.floor, 0)
        self.assertEqual(node.position, (10, 20, 0))
        self.assertEqual(node.g_cost, 0.0)
        self.assertEqual(node.h_cost, 0.0)
        self.assertEqual(node.f_cost, 0.0)
        self.assertIsNone(node.parent)
        self.assertEqual(node.movement_type, MovementType.WALK)
    
    def test_a_star_algorithm(self):
        """Test A* pathfinding algorithm"""
        # Set algorithm to A*
        self.pathfinder.algorithm = PathfindingAlgorithm.A_STAR
        
        goal = NavigationGoal(target_x=90, target_y=90)
        result = self.pathfinder.find_path(73, 75, goal, self.test_analysis)
        
        if result.success:
            self.assertGreater(len(result.path), 0)
            self.assertEqual(result.algorithm_used, PathfindingAlgorithm.A_STAR)
            
            # Check path continuity
            for i in range(1, len(result.path)):
                prev_node = result.path[i-1]
                curr_node = result.path[i]
                
                # Adjacent nodes should be close
                dx = abs(curr_node.x - prev_node.x)
                dy = abs(curr_node.y - prev_node.y)
                self.assertLessEqual(max(dx, dy), 1)
    
    def test_dijkstra_algorithm(self):
        """Test Dijkstra pathfinding algorithm"""
        # Set algorithm to Dijkstra
        self.pathfinder.algorithm = PathfindingAlgorithm.DIJKSTRA
        
        goal = NavigationGoal(target_x=90, target_y=90)
        result = self.pathfinder.find_path(73, 75, goal, self.test_analysis)
        
        if result.success:
            self.assertGreater(len(result.path), 0)
            self.assertEqual(result.algorithm_used, PathfindingAlgorithm.DIJKSTRA)
    
    def test_path_caching(self):
        """Test path caching mechanism"""
        goal = NavigationGoal(target_x=85, target_y=85)
        
        # First call - should calculate path
        result1 = self.pathfinder.find_path(75, 75, goal, self.test_analysis)
        cache_misses_1 = self.pathfinder.stats['cache_misses']
        
        # Second call - should use cache
        result2 = self.pathfinder.find_path(75, 75, goal, self.test_analysis)
        cache_hits_1 = self.pathfinder.stats['cache_hits']
        
        # Cache hits should increase
        self.assertGreater(cache_hits_1, 0)
    
    def test_danger_level_calculation(self):
        """Test danger level calculation"""
        # Create node in safe area
        safe_node = PathNode(73, 75, 0)
        danger_safe = self.pathfinder._calculate_danger_level(safe_node, self.test_analysis)
        
        self.assertIsInstance(danger_safe, float)
        self.assertGreaterEqual(danger_safe, 0.0)
        self.assertLessEqual(danger_safe, 1.0)
    
    def test_movement_cost_calculation(self):
        """Test movement cost calculation"""
        from_node = PathNode(70, 70, 0)
        to_node = PathNode(71, 71, 0)
        goal = NavigationGoal(target_x=80, target_y=80)
        
        cost = self.pathfinder._movement_cost(from_node, to_node, goal)
        
        self.assertIsInstance(cost, float)
        self.assertGreater(cost, 0)
    
    def test_heuristic_cost(self):
        """Test heuristic cost calculation"""
        node = PathNode(70, 70, 0)
        goal_node = PathNode(80, 80, 0)
        
        h_cost = self.pathfinder._heuristic_cost(node, goal_node)
        
        self.assertIsInstance(h_cost, float)
        self.assertGreater(h_cost, 0)
        
        # Heuristic should be consistent
        h_cost_reverse = self.pathfinder._heuristic_cost(goal_node, node)
        self.assertEqual(h_cost, h_cost_reverse)
    
    def test_neighbor_generation(self):
        """Test neighbor node generation"""
        node = PathNode(75, 75, 0)
        goal = NavigationGoal(target_x=80, target_y=80)
        
        neighbors = self.pathfinder._get_neighbors(node, goal, self.test_analysis)
        
        self.assertIsInstance(neighbors, list)
        # Should have some neighbors (8-directional movement)
        self.assertGreater(len(neighbors), 0)
        self.assertLessEqual(len(neighbors), 8)  # Basic 8-directional
        
        for neighbor in neighbors:
            self.assertIsInstance(neighbor, PathNode)
            # Neighbors should be adjacent
            dx = abs(neighbor.x - node.x)
            dy = abs(neighbor.y - node.y)
            self.assertLessEqual(max(dx, dy), 1)
    
    def test_pathfinding_statistics(self):
        """Test pathfinding statistics"""
        goal = NavigationGoal(target_x=80, target_y=80)
        
        # Perform several pathfinding operations
        for _ in range(3):
            self.pathfinder.find_path(73, 75, goal, self.test_analysis)
        
        stats = self.pathfinder.get_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('paths_calculated', stats)
        self.assertIn('avg_calculation_time', stats)
        self.assertIn('cache_hits', stats)
        self.assertIn('cache_misses', stats)
        self.assertIn('successful_navigations', stats)
        
        self.assertGreaterEqual(stats['paths_calculated'], 3)
        self.assertGreater(stats['avg_calculation_time'], 0)
    
    def test_cache_management(self):
        """Test cache management"""
        # Clear cache
        self.pathfinder.clear_cache()
        
        # Cache should be empty
        self.assertEqual(len(self.pathfinder.path_cache), 0)
        
        # Add some paths to cache
        goal = NavigationGoal(target_x=80, target_y=80)
        self.pathfinder.find_path(73, 75, goal, self.test_analysis)
        
        # Cache should have entries
        self.assertGreater(len(self.pathfinder.path_cache), 0)
    
    def test_error_handling_pathfinding(self):
        """Test error handling in pathfinding"""
        # Test with invalid goal
        invalid_goal = NavigationGoal(target_x=-1000, target_y=-1000)
        
        with self.assertLogs(level='WARNING'):
            result = self.pathfinder.find_path(73, 75, invalid_goal, self.test_analysis)
            self.assertFalse(result.success)
    
    def test_path_result_properties(self):
        """Test path result properties"""
        goal = NavigationGoal(target_x=80, target_y=80)
        result = self.pathfinder.find_path(73, 75, goal, self.test_analysis)
        
        # Check all required properties exist
        self.assertIsInstance(result.path, list)
        self.assertIsInstance(result.total_cost, float)
        self.assertIsInstance(result.algorithm_used, PathfindingAlgorithm)
        self.assertIsInstance(result.calculation_time, float)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.distance, float)
        self.assertIsInstance(result.estimated_time, float)
        self.assertIsInstance(result.danger_rating, float)
        self.assertIsInstance(result.teleports_used, int)
        self.assertIsInstance(result.alternative_paths, list)
        
        # Validate ranges
        self.assertGreaterEqual(result.danger_rating, 0.0)
        self.assertLessEqual(result.danger_rating, 1.0)
        self.assertGreaterEqual(result.teleports_used, 0)


@unittest.skipUnless(NAVIGATION_AVAILABLE, "Navigation system not available")
class TestNavigationPerformance(unittest.TestCase):
    """Performance tests for navigation system"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.analyzer = AdvancedMinimapAnalyzer(device="cpu", enable_yolo=False)
        self.pathfinder = OSRSPathfinder(self.analyzer)
        self.test_screenshot = create_test_game_screenshot()
    
    def tearDown(self):
        """Clean up after performance tests"""
        if hasattr(self, 'pathfinder'):
            self.pathfinder.cleanup()
        if hasattr(self, 'analyzer'):
            self.analyzer.cleanup()
    
    def test_minimap_analysis_performance(self):
        """Test minimap analysis performance"""
        iterations = 50
        start_time = time.time()
        
        for _ in range(iterations):
            result = self.analyzer.analyze_minimap(self.test_screenshot)
            self.assertIsInstance(result, MinimapAnalysisResult)
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        fps = iterations / total_time
        
        # Performance requirements
        self.assertLess(avg_time, 0.1)  # Less than 100ms per frame
        self.assertGreater(fps, 10)     # At least 10 FPS
        
        print(f"Minimap Analysis Performance:")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  FPS: {fps:.1f}")
    
    def test_pathfinding_performance(self):
        """Test pathfinding performance"""
        test_analysis = self.analyzer.analyze_minimap(self.test_screenshot)
        iterations = 20
        
        start_time = time.time()
        
        for i in range(iterations):
            goal = NavigationGoal(
                target_x=80 + (i % 20),
                target_y=80 + (i % 20)
            )
            result = self.pathfinder.find_path(73, 75, goal, test_analysis)
            self.assertIsInstance(result, PathResult)
        
        total_time = time.time() - start_time
        avg_time = total_time / iterations
        
        # Performance requirements
        self.assertLess(avg_time, 1.0)  # Less than 1 second per path
        
        print(f"Pathfinding Performance:")
        print(f"  Average time: {avg_time*1000:.2f}ms")
        print(f"  Paths per second: {1/avg_time:.1f}")
    
    def test_memory_usage(self):
        """Test memory usage and cleanup"""
        import gc
        import sys
        
        # Force garbage collection
        gc.collect()
        
        # Get initial memory usage (approximate)
        initial_objects = len(gc.get_objects())
        
        # Create and destroy many analyzers
        for _ in range(10):
            analyzer = AdvancedMinimapAnalyzer(device="cpu", enable_yolo=False)
            pathfinder = OSRSPathfinder(analyzer)
            
            # Use them
            result = analyzer.analyze_minimap(self.test_screenshot)
            goal = NavigationGoal(target_x=80, target_y=80)
            path_result = pathfinder.find_path(73, 75, goal, result)
            
            # Clean up
            pathfinder.cleanup()
            analyzer.cleanup()
            del pathfinder
            del analyzer
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        final_objects = len(gc.get_objects())
        
        # Should not have excessive memory growth
        object_growth = final_objects - initial_objects
        self.assertLess(object_growth, 1000)  # Reasonable growth limit
        
        print(f"Memory Usage:")
        print(f"  Object growth: {object_growth}")
    
    def test_concurrent_analysis(self):
        """Test concurrent minimap analysis"""
        import concurrent.futures
        
        def analyze_minimap():
            return self.analyzer.analyze_minimap(self.test_screenshot)
        
        # Run concurrent analysis
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(analyze_minimap) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All should succeed
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertIsInstance(result, MinimapAnalysisResult)
    
    def test_stress_test(self):
        """Stress test for navigation system"""
        # Run many operations in sequence
        for i in range(100):
            # Analyze minimap
            result = self.analyzer.analyze_minimap(self.test_screenshot)
            
            # Calculate path every 10th iteration
            if i % 10 == 0:
                goal = NavigationGoal(
                    target_x=70 + (i % 30),
                    target_y=70 + (i % 30)
                )
                path_result = self.pathfinder.find_path(73, 75, goal, result)
                self.assertIsInstance(path_result, PathResult)
        
        # System should still be responsive
        stats = self.analyzer.get_performance_stats()
        self.assertGreater(stats['frames_processed'], 100)


class TestNavigationIntegration(unittest.TestCase):
    """Integration tests for navigation system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        if NAVIGATION_AVAILABLE:
            self.analyzer = AdvancedMinimapAnalyzer(device="cpu", enable_yolo=False)
            self.pathfinder = OSRSPathfinder(self.analyzer)
            self.test_screenshot = create_test_game_screenshot()
    
    def tearDown(self):
        """Clean up after integration tests"""
        if NAVIGATION_AVAILABLE:
            if hasattr(self, 'pathfinder'):
                self.pathfinder.cleanup()
            if hasattr(self, 'analyzer'):
                self.analyzer.cleanup()
    
    @unittest.skipUnless(NAVIGATION_AVAILABLE, "Navigation system not available")
    def test_full_navigation_pipeline(self):
        """Test complete navigation pipeline"""
        # 1. Analyze minimap
        analysis = self.analyzer.analyze_minimap(self.test_screenshot)
        self.assertIsInstance(analysis, MinimapAnalysisResult)
        
        # 2. Create navigation goal
        goal = NavigationGoal(
            target_x=90,
            target_y=90,
            max_danger=0.5,
            prefer_safe_route=True,
            allow_teleports=True
        )
        
        # 3. Find path
        player_x, player_y = analysis.player_position
        path_result = self.pathfinder.find_path(player_x, player_y, goal, analysis)
        
        # 4. Validate results
        self.assertIsInstance(path_result, PathResult)
        
        if path_result.success:
            # Verify path makes sense
            self.assertGreater(len(path_result.path), 0)
            
            # First node should be near starting position
            start_node = path_result.path[0]
            self.assertLessEqual(abs(start_node.x - player_x), 2)
            self.assertLessEqual(abs(start_node.y - player_y), 2)
            
            # Last node should be near goal
            end_node = path_result.path[-1]
            self.assertLessEqual(abs(end_node.x - goal.target_x), 2)
            self.assertLessEqual(abs(end_node.y - goal.target_y), 2)
    
    @unittest.skipUnless(NAVIGATION_AVAILABLE, "Navigation system not available")
    def test_real_time_simulation(self):
        """Test real-time navigation simulation"""
        analysis_times = []
        pathfinding_times = []
        
        # Simulate real-time updates
        for frame in range(30):  # 30 frames
            start_time = time.time()
            
            # Analyze current frame
            analysis = self.analyzer.analyze_minimap(self.test_screenshot)
            analysis_time = time.time() - start_time
            analysis_times.append(analysis_time)
            
            # Calculate path every 10 frames
            if frame % 10 == 0:
                path_start = time.time()
                goal = NavigationGoal(
                    target_x=80 + (frame % 20),
                    target_y=80 + (frame % 20)
                )
                player_x, player_y = analysis.player_position
                path_result = self.pathfinder.find_path(player_x, player_y, goal, analysis)
                path_time = time.time() - path_start
                pathfinding_times.append(path_time)
            
            # Simulate frame rate (60 FPS = 16.67ms per frame)
            frame_time = time.time() - start_time
            if frame_time < 0.0167:
                time.sleep(0.0167 - frame_time)
        
        # Verify performance
        avg_analysis_time = sum(analysis_times) / len(analysis_times)
        avg_pathfinding_time = sum(pathfinding_times) / len(pathfinding_times) if pathfinding_times else 0
        
        self.assertLess(avg_analysis_time, 0.0167)  # Should be faster than 60 FPS
        self.assertLess(avg_pathfinding_time, 0.1)   # Should be faster than 100ms
        
        print(f"Real-time Performance:")
        print(f"  Average analysis time: {avg_analysis_time*1000:.2f}ms")
        print(f"  Average pathfinding time: {avg_pathfinding_time*1000:.2f}ms")


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    if NAVIGATION_AVAILABLE:
        test_suite.addTest(unittest.makeSuite(TestAdvancedMinimapAnalyzer))
        test_suite.addTest(unittest.makeSuite(TestOSRSPathfinder))
        test_suite.addTest(unittest.makeSuite(TestNavigationPerformance))
        test_suite.addTest(unittest.makeSuite(TestNavigationIntegration))
    else:
        print("WARNING: Navigation system not available, skipping tests")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Results:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")