#!/usr/bin/env python3
"""
OSRS Bot Framework - Behavior Modeling Tests

Comprehensive tests for the AI behavior modeling and advanced anti-detection systems.
Tests both AI-powered features and fallback mechanisms.
"""

import sys
import unittest
import time
import random
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from safety.behavior_modeling import (
    HumanBehaviorModel, MouseMovement, ActionEvent, BehaviorProfile,
    BehaviorAnalyzer, PatternDetector, AnomalyDetector,
    get_behavior_model, initialize_behavior_modeling
)
from safety.advanced_anti_detection import (
    AdvancedAntiDetection, BehavioralObfuscator, PatternBreaker,
    RiskPredictor, FingerprintManager, DetectionEvent, SafetyRecommendation,
    get_anti_detection_system, initialize_advanced_anti_detection
)


class TestBehaviorModeling(unittest.TestCase):
    """Test cases for the behavior modeling system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.behavior_model = HumanBehaviorModel()
        
    def tearDown(self):
        """Clean up after tests"""
        pass
    
    def test_behavior_model_initialization(self):
        """Test behavior model initializes correctly"""
        self.assertIsNotNone(self.behavior_model)
        self.assertIsNotNone(self.behavior_model.mouse_movements)
        self.assertIsNotNone(self.behavior_model.action_events)
        self.assertIsNotNone(self.behavior_model.behavior_analyzer)
        self.assertIsNotNone(self.behavior_model.pattern_detector)
        self.assertIsNotNone(self.behavior_model.anomaly_detector)
    
    def test_mouse_movement_recording(self):
        """Test mouse movement recording"""
        start_pos = (100, 100)
        end_pos = (200, 200)
        duration = 0.5
        
        # Record movement
        self.behavior_model.record_mouse_movement(start_pos, end_pos, duration)
        
        # Check movement was recorded
        self.assertEqual(len(self.behavior_model.mouse_movements), 1)
        
        movement = self.behavior_model.mouse_movements[0]
        self.assertEqual(movement.start_pos, start_pos)
        self.assertEqual(movement.end_pos, end_pos)
        self.assertEqual(movement.duration, duration)
        self.assertGreater(movement.timestamp, 0)
        self.assertGreaterEqual(movement.acceleration, 0)
        self.assertGreaterEqual(movement.jitter, 0)
    
    def test_action_recording(self):
        """Test action event recording"""
        action_type = "click"
        position = (150, 150)
        duration = 0.1
        context = {"button": "left"}
        
        # Record action
        self.behavior_model.record_action(action_type, position, duration, context)
        
        # Check action was recorded
        self.assertEqual(len(self.behavior_model.action_events), 1)
        
        action = self.behavior_model.action_events[0]
        self.assertEqual(action.action_type, action_type)
        self.assertEqual(action.position, position)
        self.assertEqual(action.duration, duration)
        self.assertEqual(action.context, context)
        self.assertGreater(action.timestamp, 0)
    
    def test_human_movement_generation(self):
        """Test human-like movement generation"""
        start_pos = (100, 100)
        end_pos = (300, 250)
        
        # Generate movement path
        path = self.behavior_model.generate_human_movement(start_pos, end_pos)
        
        # Check path is valid
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)
        
        # Check start and end points
        if len(path) > 0:
            # Path should have reasonable progression
            for point in path:
                self.assertIsInstance(point, tuple)
                self.assertEqual(len(point), 2)
                self.assertIsInstance(point[0], int)
                self.assertIsInstance(point[1], int)
    
    def test_action_timing_prediction(self):
        """Test action timing prediction"""
        action_types = ["click", "double_click", "drag", "key_press", "type", "scroll"]
        
        for action_type in action_types:
            timing = self.behavior_model.get_next_action_timing(action_type)
            
            # Check timing is reasonable
            self.assertIsInstance(timing, float)
            self.assertGreater(timing, 0)
            self.assertLess(timing, 5.0)  # Should be less than 5 seconds
    
    def test_break_recommendation(self):
        """Test break recommendation system"""
        # Initially should not recommend break
        should_break, duration = self.behavior_model.should_take_break()
        
        self.assertIsInstance(should_break, bool)
        self.assertIsInstance(duration, float)
        self.assertGreaterEqual(duration, 0)
    
    def test_risk_assessment(self):
        """Test risk assessment functionality"""
        risk_assessment = self.behavior_model.get_risk_assessment()
        
        # Check risk assessment structure
        self.assertIsInstance(risk_assessment, dict)
        self.assertIn('overall_risk', risk_assessment)
        self.assertIn('detection_risk', risk_assessment)
        self.assertIn('pattern_risk', risk_assessment)
        self.assertIn('timing_risk', risk_assessment)
        
        # Check risk values are in valid range
        for risk_type, risk_value in risk_assessment.items():
            self.assertIsInstance(risk_value, float)
            self.assertGreaterEqual(risk_value, 0.0)
            self.assertLessEqual(risk_value, 1.0)
    
    def test_behavior_profile_update(self):
        """Test behavior profile updating"""
        # Add some movement data
        for i in range(20):
            start = (random.randint(0, 800), random.randint(0, 600))
            end = (random.randint(0, 800), random.randint(0, 600))
            duration = random.uniform(0.1, 0.5)
            self.behavior_model.record_mouse_movement(start, end, duration)
        
        # Update profile
        profile = self.behavior_model.update_behavior_profile()
        
        if profile:  # May be None if insufficient data
            self.assertIsInstance(profile, BehaviorProfile)
            self.assertIsInstance(profile.mouse_speed_avg, float)
            self.assertIsInstance(profile.mouse_speed_std, float)
            self.assertIsInstance(profile.reaction_time_avg, float)
            self.assertIsInstance(profile.reaction_time_std, float)
            self.assertIsInstance(profile.consistency_score, float)
            
            # Check values are in reasonable ranges
            self.assertGreaterEqual(profile.consistency_score, 0.0)
            self.assertLessEqual(profile.consistency_score, 1.0)
    
    def test_behavior_analyzer(self):
        """Test behavior analyzer functionality"""
        analyzer = BehaviorAnalyzer()
        
        # Test movement analysis
        speeds = [100.0, 120.0, 90.0, 110.0, 105.0]
        accelerations = [10.0, 12.0, 8.0, 11.0, 9.0]
        
        analyzer.update_movement_analysis(speeds, accelerations)
        
        # Check patterns were updated
        self.assertIn('speed_mean', analyzer.patterns)
        self.assertIn('speed_std', analyzer.patterns)
    
    def test_pattern_detector(self):
        """Test pattern detection functionality"""
        detector = PatternDetector()
        
        # Test with repetitive data
        data = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        patterns = detector.detect_patterns(data)
        
        self.assertIsInstance(patterns, list)
    
    def test_anomaly_detector(self):
        """Test anomaly detection functionality"""
        detector = AnomalyDetector()
        
        # Create test movements
        movements = []
        for i in range(10):
            movement = MouseMovement(
                start_pos=(i*10, i*10),
                end_pos=(i*10+50, i*10+50),
                duration=0.1 + i*0.01,
                timestamp=time.time() + i
            )
            movements.append(movement)
        
        # Add an anomalous movement
        anomalous_movement = MouseMovement(
            start_pos=(0, 0),
            end_pos=(1000, 1000),  # Very fast movement
            duration=0.001,  # Very short duration
            timestamp=time.time()
        )
        movements.append(anomalous_movement)
        
        anomalies = detector.detect_movement_anomalies(movements)
        
        self.assertIsInstance(anomalies, list)
        # Should detect the anomalous movement
        self.assertGreater(len(anomalies), 0)


class TestAdvancedAntiDetection(unittest.TestCase):
    """Test cases for the advanced anti-detection system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.anti_detection = AdvancedAntiDetection()
    
    def tearDown(self):
        """Clean up after tests"""
        self.anti_detection.stop_monitoring()
    
    def test_anti_detection_initialization(self):
        """Test anti-detection system initializes correctly"""
        self.assertIsNotNone(self.anti_detection)
        self.assertIsNotNone(self.anti_detection.behavioral_obfuscator)
        self.assertIsNotNone(self.anti_detection.pattern_breaker)
        self.assertIsNotNone(self.anti_detection.risk_predictor)
        self.assertIsNotNone(self.anti_detection.fingerprint_manager)
        self.assertIsInstance(self.anti_detection.risk_thresholds, dict)
        self.assertIsInstance(self.anti_detection.stats, dict)
    
    def test_risk_assessment(self):
        """Test risk assessment functionality"""
        risk_assessment = self.anti_detection.assess_current_risk()
        
        # Check risk assessment structure
        self.assertIsInstance(risk_assessment, dict)
        self.assertIn('overall_risk', risk_assessment)
        self.assertIn('behavioral_risk', risk_assessment)
        self.assertIn('pattern_risk', risk_assessment)
        self.assertIn('fingerprint_risk', risk_assessment)
        
        # Check risk values are in valid range
        for risk_type, risk_value in risk_assessment.items():
            self.assertIsInstance(risk_value, float)
            self.assertGreaterEqual(risk_value, 0.0)
            self.assertLessEqual(risk_value, 1.0)
    
    def test_safe_action_timing(self):
        """Test safe action timing generation"""
        action_types = ["click", "double_click", "drag", "key_press"]
        
        for action_type in action_types:
            safe_timing = self.anti_detection.get_safe_action_timing(action_type)
            
            self.assertIsInstance(safe_timing, float)
            self.assertGreater(safe_timing, 0)
            self.assertLess(safe_timing, 10.0)  # Should be reasonable
    
    def test_safe_mouse_path(self):
        """Test safe mouse path generation"""
        start_pos = (100, 100)
        end_pos = (200, 200)
        
        safe_path = self.anti_detection.get_safe_mouse_path(start_pos, end_pos)
        
        self.assertIsInstance(safe_path, list)
        self.assertGreater(len(safe_path), 0)
        
        # Check path points are valid
        for point in safe_path:
            self.assertIsInstance(point, tuple)
            self.assertEqual(len(point), 2)
            self.assertIsInstance(point[0], int)
            self.assertIsInstance(point[1], int)
    
    def test_break_recommendation(self):
        """Test break recommendation system"""
        should_break, duration, reason = self.anti_detection.check_break_recommendation()
        
        self.assertIsInstance(should_break, bool)
        self.assertIsInstance(duration, float)
        self.assertIsInstance(reason, str)
        self.assertGreaterEqual(duration, 0)
    
    def test_safety_recommendations(self):
        """Test safety recommendations"""
        recommendations = self.anti_detection.get_safety_recommendations()
        
        self.assertIsInstance(recommendations, list)
        
        for rec in recommendations:
            self.assertIsInstance(rec, SafetyRecommendation)
            self.assertIsInstance(rec.recommendation_type, str)
            self.assertIsInstance(rec.urgency, float)
            self.assertIsInstance(rec.description, str)
            self.assertIsInstance(rec.action_required, bool)
            self.assertIsInstance(rec.estimated_benefit, float)
            
            # Check urgency is in valid range
            self.assertGreaterEqual(rec.urgency, 0.0)
            self.assertLessEqual(rec.urgency, 1.0)
    
    def test_monitoring_system(self):
        """Test monitoring system"""
        # Test start monitoring
        self.anti_detection.start_monitoring()
        self.assertTrue(self.anti_detection.monitoring_active)
        
        # Test stop monitoring
        self.anti_detection.stop_monitoring()
        self.assertFalse(self.anti_detection.monitoring_active)
    
    def test_statistics(self):
        """Test statistics retrieval"""
        stats = self.anti_detection.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('monitoring_active', stats)
        self.assertIn('current_risk', stats)
        self.assertIn('actions_taken', stats)
        self.assertIn('risks_mitigated', stats)
        self.assertIn('patterns_broken', stats)
        self.assertIn('fingerprints_rotated', stats)


class TestBehavioralObfuscator(unittest.TestCase):
    """Test cases for the behavioral obfuscator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.obfuscator = BehavioralObfuscator()
    
    def test_timing_obfuscation(self):
        """Test timing obfuscation"""
        base_timing = 0.1
        action_type = "click"
        
        obfuscated_timing = self.obfuscator.obfuscate_timing(base_timing, action_type)
        
        self.assertIsInstance(obfuscated_timing, float)
        self.assertGreater(obfuscated_timing, 0)
        # Should be somewhat close to base timing but with variation
        self.assertLess(abs(obfuscated_timing - base_timing), 1.0)
    
    def test_mouse_path_obfuscation(self):
        """Test mouse path obfuscation"""
        base_path = [(100, 100), (120, 110), (140, 120), (160, 130), (180, 140)]
        
        obfuscated_path = self.obfuscator.obfuscate_mouse_path(base_path)
        
        self.assertIsInstance(obfuscated_path, list)
        self.assertEqual(len(obfuscated_path), len(base_path))
        
        # Check points are still reasonable
        for i, (orig, obf) in enumerate(zip(base_path, obfuscated_path)):
            self.assertIsInstance(obf, tuple)
            self.assertEqual(len(obf), 2)
            # Should be close to original but with some variation
            self.assertLess(abs(obf[0] - orig[0]), 20)
            self.assertLess(abs(obf[1] - orig[1]), 20)
    
    def test_obfuscation_parameter_updates(self):
        """Test obfuscation parameter updates"""
        initial_level = self.obfuscator.obfuscation_level
        
        self.obfuscator.update_obfuscation_parameters()
        
        # Level should be updated (may be same due to time-based variation)
        self.assertIsInstance(self.obfuscator.obfuscation_level, float)
        self.assertGreaterEqual(self.obfuscator.obfuscation_level, 0.0)
        self.assertLessEqual(self.obfuscator.obfuscation_level, 1.0)


class TestPatternBreaker(unittest.TestCase):
    """Test cases for the pattern breaker"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pattern_breaker = PatternBreaker()
    
    def test_pattern_risk_calculation(self):
        """Test pattern risk calculation"""
        # Add some patterns
        for i in range(10):
            self.pattern_breaker.record_action("click", f"context_{i % 3}")
        
        risk = self.pattern_breaker.calculate_pattern_risk()
        
        self.assertIsInstance(risk, float)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
    
    def test_pattern_breaking_decision(self):
        """Test pattern breaking decision"""
        should_break = self.pattern_breaker.should_break_pattern()
        
        self.assertIsInstance(should_break, bool)
    
    def test_pattern_recording(self):
        """Test pattern recording"""
        initial_count = len(self.pattern_breaker.pattern_history)
        
        self.pattern_breaker.record_action("click", "test_context")
        
        self.assertEqual(len(self.pattern_breaker.pattern_history), initial_count + 1)
        self.assertIn("click:test_context", self.pattern_breaker.pattern_history)


class TestRiskPredictor(unittest.TestCase):
    """Test cases for the risk predictor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_predictor = RiskPredictor()
    
    def test_risk_prediction(self):
        """Test risk prediction functionality"""
        # Add some risk history
        for i in range(20):
            self.risk_predictor.record_risk_level(0.3 + (i * 0.01))
        
        predictions = self.risk_predictor.predict_future_risk()
        
        self.assertIsInstance(predictions, dict)
        self.assertIn('short_term', predictions)
        self.assertIn('medium_term', predictions)
        self.assertIn('long_term', predictions)
        self.assertIn('trend', predictions)
        
        # Check prediction values are in valid range
        for key, value in predictions.items():
            if key != 'trend':  # Trend can be negative
                self.assertIsInstance(value, float)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


class TestFingerprintManager(unittest.TestCase):
    """Test cases for the fingerprint manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fingerprint_manager = FingerprintManager()
    
    def test_fingerprint_generation(self):
        """Test fingerprint generation"""
        fingerprint = self.fingerprint_manager.get_current_fingerprint()
        
        self.assertIsInstance(fingerprint, dict)
        self.assertIn('mouse_speed_profile', fingerprint)
        self.assertIn('timing_preference', fingerprint)
        self.assertIn('break_tendency', fingerprint)
        self.assertIn('accuracy_level', fingerprint)
        self.assertIn('consistency_factor', fingerprint)
        self.assertIn('generated_at', fingerprint)
        self.assertIn('signature', fingerprint)
    
    def test_fingerprint_rotation(self):
        """Test fingerprint rotation"""
        old_fingerprint = self.fingerprint_manager.get_current_fingerprint()
        
        self.fingerprint_manager.rotate_fingerprint()
        
        new_fingerprint = self.fingerprint_manager.get_current_fingerprint()
        
        # Fingerprints should be different
        self.assertNotEqual(old_fingerprint['signature'], new_fingerprint['signature'])
    
    def test_rotation_decision(self):
        """Test rotation decision logic"""
        should_rotate = self.fingerprint_manager.should_rotate_fingerprint()
        
        self.assertIsInstance(should_rotate, bool)
    
    def test_fingerprint_risk_calculation(self):
        """Test fingerprint risk calculation"""
        risk = self.fingerprint_manager.calculate_fingerprint_risk()
        
        self.assertIsInstance(risk, float)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the entire safety system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.behavior_model = get_behavior_model()
        self.anti_detection = get_anti_detection_system()
    
    def test_system_integration(self):
        """Test integration between behavior modeling and anti-detection"""
        # Record some behavior data
        for i in range(10):
            start = (random.randint(0, 800), random.randint(0, 600))
            end = (random.randint(0, 800), random.randint(0, 600))
            duration = random.uniform(0.1, 0.5)
            
            self.behavior_model.record_mouse_movement(start, end, duration)
            self.behavior_model.record_action("click", ((start[0] + end[0])//2, (start[1] + end[1])//2), 0.1)
        
        # Test that anti-detection can access behavior data
        risk_assessment = self.anti_detection.assess_current_risk()
        self.assertIsInstance(risk_assessment, dict)
        
        # Test safe action generation
        safe_timing = self.anti_detection.get_safe_action_timing("click")
        self.assertGreater(safe_timing, 0)
        
        safe_path = self.anti_detection.get_safe_mouse_path((100, 100), (200, 200))
        self.assertGreater(len(safe_path), 0)
    
    def test_global_instances(self):
        """Test global instance management"""
        # Test behavior model singleton
        model1 = get_behavior_model()
        model2 = get_behavior_model()
        self.assertIs(model1, model2)
        
        # Test anti-detection singleton
        system1 = get_anti_detection_system()
        system2 = get_anti_detection_system()
        self.assertIs(system1, system2)
    
    def test_initialization_functions(self):
        """Test initialization functions"""
        # Test behavior modeling initialization
        model = initialize_behavior_modeling()
        self.assertIsInstance(model, HumanBehaviorModel)
        
        # Test anti-detection initialization
        system = initialize_advanced_anti_detection()
        self.assertIsInstance(system, AdvancedAntiDetection)


class TestMockGPUEnvironment(unittest.TestCase):
    """Test behavior in environments without GPU support"""
    
    @patch('safety.behavior_modeling.GPU_AVAILABLE', False)
    @patch('safety.behavior_modeling.torch', None)
    def test_fallback_behavior_without_gpu(self):
        """Test that system works without GPU/PyTorch"""
        behavior_model = HumanBehaviorModel()
        
        # Should still work with fallback methods
        path = behavior_model.generate_human_movement((100, 100), (200, 200))
        self.assertIsInstance(path, list)
        self.assertGreater(len(path), 0)
        
        timing = behavior_model.get_next_action_timing("click")
        self.assertIsInstance(timing, float)
        self.assertGreater(timing, 0)
        
        risk = behavior_model.get_risk_assessment()
        self.assertIsInstance(risk, dict)


if __name__ == '__main__':
    # Run specific test categories
    import argparse
    
    parser = argparse.ArgumentParser(description='Run behavior modeling tests')
    parser.add_argument('--category', choices=['behavior', 'anti_detection', 'integration', 'all'], 
                       default='all', help='Test category to run')
    args = parser.parse_args()
    
    if args.category == 'behavior':
        # Run only behavior modeling tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestBehaviorModeling)
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBehavioralObfuscator))
    elif args.category == 'anti_detection':
        # Run only anti-detection tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestAdvancedAntiDetection)
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPatternBreaker))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRiskPredictor))
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestFingerprintManager))
    elif args.category == 'integration':
        # Run only integration tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMockGPUEnvironment))
    else:
        # Run all tests
        suite = unittest.TestLoader().discover('.')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with error code if tests failed
    sys.exit(len(result.failures) + len(result.errors))