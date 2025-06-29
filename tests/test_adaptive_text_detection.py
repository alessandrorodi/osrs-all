#!/usr/bin/env python3
"""
Unit Tests for Adaptive Text Detection System
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the classes we're testing
from vision.adaptive_text_detection import AdaptiveTextDetector, DetectedTextArea


class TestAdaptiveTextDetector(unittest.TestCase):
    """Test cases for AdaptiveTextDetector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = AdaptiveTextDetector()
        
        # Create a mock screenshot
        self.mock_screenshot = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Create mock text detections
        self.mock_detection = Mock()
        self.mock_detection.text = "Test message"
        self.mock_detection.confidence = 0.9
        self.mock_detection.x = 100
        self.mock_detection.y = 200
        self.mock_detection.width = 50
        self.mock_detection.height = 20
    
    @patch('vision.adaptive_text_detection.OCRDetector')
    def test_detector_initialization(self, mock_ocr_class):
        """Test that detector initializes properly"""
        detector = AdaptiveTextDetector()
        
        # Should create OCR detector
        mock_ocr_class.assert_called_once()
        
        # Should have UI colors defined
        self.assertIn('chat_background', detector.ui_colors)
        self.assertIn('health_red', detector.ui_colors)
    
    def test_text_content_classification(self):
        """Test text content classification logic"""
        
        # Test chat classification
        chat_texts = ["Player1: hello", "says something", "whispers quietly"]
        for text in chat_texts:
            result = self.detector._classify_text_content([text])
            self.assertEqual(result, 'chat')
        
        # Test numbers classification
        number_texts = ["123", "456hp", "99"]
        for text in number_texts:
            result = self.detector._classify_text_content([text])
            self.assertEqual(result, 'numbers')
        
        # Test interface classification
        interface_texts = ["Click here", "Attack goblin", "Use item"]
        for text in interface_texts:
            result = self.detector._classify_text_content([text])
            self.assertEqual(result, 'interface')
        
        # Test items classification
        item_texts = ["Dragon sword", "Rune plate", "Food"]
        for text in item_texts:
            result = self.detector._classify_text_content([text])
            self.assertEqual(result, 'items')
    
    def test_text_area_grouping(self):
        """Test that nearby text detections are grouped properly"""
        
        # Create mock detections that should be grouped
        detection1 = Mock()
        detection1.x, detection1.y = 100, 100
        detection1.width, detection1.height = 50, 20
        detection1.text = "Line 1"
        detection1.confidence = 0.9
        
        detection2 = Mock()
        detection2.x, detection2.y = 105, 125  # Close to detection1
        detection2.width, detection2.height = 60, 20
        detection2.text = "Line 2"
        detection2.confidence = 0.8
        
        detection3 = Mock()
        detection3.x, detection3.y = 400, 400  # Far from others
        detection3.width, detection3.height = 40, 15
        detection3.text = "Far away"
        detection3.confidence = 0.7
        
        detections = [detection1, detection2, detection3]
        
        # Group the detections
        areas = self.detector._group_text_by_areas(detections, self.mock_screenshot)
        
        # Should create 2 areas (first two grouped, third separate)
        self.assertEqual(len(areas), 2)
        
        # First area should contain 2 texts
        area1 = next(area for area in areas if len(area.texts) == 2)
        self.assertEqual(len(area1.texts), 2)
        self.assertIn("Line 1", area1.texts)
        self.assertIn("Line 2", area1.texts)
        
        # Second area should contain 1 text
        area2 = next(area for area in areas if len(area.texts) == 1)
        self.assertEqual(len(area2.texts), 1)
        self.assertIn("Far away", area2.texts)
    
    def test_distance_calculation(self):
        """Test text detection distance calculation"""
        
        # Mock detection
        detection = Mock()
        detection.x, detection.y = 150, 250
        detection.width, detection.height = 50, 20
        
        # Area bounding box [x1, y1, x2, y2]
        close_area = [140, 240, 200, 270]  # Close
        far_area = [500, 500, 600, 600]     # Far
        
        # Test close distance
        is_far_close = self.detector._is_far_from_area(detection, close_area, threshold=100)
        self.assertFalse(is_far_close)
        
        # Test far distance
        is_far_far = self.detector._is_far_from_area(detection, far_area, threshold=100)
        self.assertTrue(is_far_far)
    
    def test_area_classification_by_position(self):
        """Test that text areas are classified by screen position"""
        
        # Create areas in different screen positions
        screenshot = np.zeros((1000, 1500, 3), dtype=np.uint8)  # 1500x1000
        
        # Bottom area (chat)
        bottom_area = DetectedTextArea(
            name="test", 
            bbox=(100, 850, 200, 50),  # y=850 > 80% of 1000
            text_type="interface",
            confidence=0.9,
            texts=["Player: hello"]
        )
        
        # Right area (inventory)
        right_area = DetectedTextArea(
            name="test",
            bbox=(1200, 300, 100, 50),  # x=1200 > 70% of 1500
            text_type="items",
            confidence=0.8,
            texts=["Dragon sword"]
        )
        
        # Top area (overlay)
        top_area = DetectedTextArea(
            name="test",
            bbox=(500, 50, 150, 30),  # y=50 < 20% of 1000
            text_type="interface",
            confidence=0.7,
            texts=["Mining: 15"]
        )
        
        areas = [bottom_area, right_area, top_area]
        classified_areas = self.detector._classify_text_areas(areas, screenshot)
        
        # Check classifications
        self.assertEqual(classified_areas[0].text_type, 'chat')    # Bottom -> chat
        self.assertEqual(classified_areas[1].text_type, 'inventory')  # Right -> inventory
        self.assertEqual(classified_areas[2].text_type, 'overlay')    # Top -> overlay
    
    def test_structured_data_extraction(self):
        """Test extraction of structured data from classified areas"""
        
        # Create test areas
        chat_area = DetectedTextArea(
            name="chat_test",
            bbox=(100, 500, 200, 50),
            text_type="chat",
            confidence=0.9,
            texts=["Player1: Hello world", "System: You gained XP"]
        )
        
        inventory_area = DetectedTextArea(
            name="inv_test", 
            bbox=(1000, 300, 100, 200),
            text_type="inventory",
            confidence=0.8,
            texts=["Dragon scimitar", "Rune armor", "Food item"]
        )
        
        numbers_area = DetectedTextArea(
            name="num_test",
            bbox=(200, 100, 50, 30),
            text_type="numbers",
            confidence=0.9,
            texts=["99", "1337", "42hp"]
        )
        
        areas = [chat_area, inventory_area, numbers_area]
        results = self.detector._extract_structured_data(areas)
        
        # Check chat messages
        self.assertEqual(len(results['chat_messages']), 2)
        self.assertTrue(any('Hello world' in msg['text'] for msg in results['chat_messages']))
        
        # Check items
        self.assertEqual(len(results['items']), 3)
        self.assertTrue(any('Dragon scimitar' in item['name'] for item in results['items']))
        
        # Check numbers
        self.assertEqual(len(results['numbers']), 3)
        self.assertTrue(any('99' in num['value'] for num in results['numbers']))
    
    @patch.object(AdaptiveTextDetector, '_group_text_by_areas')
    @patch.object(AdaptiveTextDetector, '_classify_text_areas')
    @patch.object(AdaptiveTextDetector, '_extract_structured_data')
    def test_detect_all_text_workflow(self, mock_extract, mock_classify, mock_group):
        """Test the complete detect_all_text workflow"""
        
        # Setup mocks
        mock_areas = [Mock()]
        mock_classified = [Mock()]
        mock_results = {'chat_messages': [], 'items': []}
        
        mock_group.return_value = mock_areas
        mock_classify.return_value = mock_classified
        mock_extract.return_value = mock_results
        
        # Mock OCR detector
        with patch.object(self.detector.ocr, 'detect_text') as mock_ocr:
            mock_ocr.return_value = [self.mock_detection]
            
            # Run detection
            results = self.detector.detect_all_text(self.mock_screenshot)
            
            # Verify workflow
            mock_ocr.assert_called_once()
            mock_group.assert_called_once()
            mock_classify.assert_called_once() 
            mock_extract.assert_called_once()
            
            # Check results structure
            self.assertIn('performance', results)
            self.assertIn('processing_time', results['performance'])
    
    def test_summary_generation(self):
        """Test summary generation from results"""
        
        test_results = {
            'chat_messages': [{'text': 'msg1'}, {'text': 'msg2'}],
            'items': [{'name': 'item1'}, {'name': 'item2'}, {'name': 'item3'}],
            'interface_elements': [{'text': 'btn1'}],
            'overlays': [],
            'numbers': [{'value': '99'}],
            'performance': {'processing_time': 2.5}
        }
        
        summary = self.detector.get_summary(test_results)
        
        self.assertEqual(summary['chat_messages'], 2)
        self.assertEqual(summary['items'], 3)
        self.assertEqual(summary['interface_elements'], 1)
        self.assertEqual(summary['overlays'], 0)
        self.assertEqual(summary['numbers'], 1)
        self.assertEqual(summary['processing_time'], 2.5)
    
    def test_error_handling(self):
        """Test error handling in detection workflow"""
        
        # Mock OCR to raise exception
        with patch.object(self.detector.ocr, 'detect_text') as mock_ocr:
            mock_ocr.side_effect = Exception("OCR failed")
            
            # Should not crash, should return default structure
            results = self.detector.detect_all_text(self.mock_screenshot)
            
            self.assertIn('chat_messages', results)
            self.assertIn('items', results)
            self.assertIn('error', results)
            self.assertEqual(len(results['chat_messages']), 0)
    
    def test_empty_detections(self):
        """Test handling of empty OCR results"""
        
        with patch.object(self.detector.ocr, 'detect_text') as mock_ocr:
            mock_ocr.return_value = []  # No detections
            
            results = self.detector.detect_all_text(self.mock_screenshot)
            
            # Should return empty but valid structure
            self.assertEqual(len(results['chat_messages']), 0)
            self.assertEqual(len(results['items']), 0)
            self.assertIn('performance', results)


class TestDetectedTextArea(unittest.TestCase):
    """Test cases for DetectedTextArea dataclass"""
    
    def test_text_area_creation(self):
        """Test creation of DetectedTextArea"""
        area = DetectedTextArea(
            name="test_area",
            bbox=(100, 200, 300, 50),
            text_type="chat",
            confidence=0.85,
            texts=["Hello", "World"]
        )
        
        self.assertEqual(area.name, "test_area")
        self.assertEqual(area.bbox, (100, 200, 300, 50))
        self.assertEqual(area.text_type, "chat")
        self.assertEqual(area.confidence, 0.85)
        self.assertEqual(len(area.texts), 2)


class TestIntegrationWithRealData(unittest.TestCase):
    """Integration tests with real-ish data"""
    
    def setUp(self):
        self.detector = AdaptiveTextDetector()
    
    def test_runelite_like_detection(self):
        """Test detection patterns similar to RuneLite"""
        
        # Mock detections that would come from RuneLite
        runelite_detections = [
            # Mining overlay
            Mock(text="Mining", x=20, y=80, width=60, height=15, confidence=0.99),
            Mock(text="Total mined: 17", x=20, y=100, width=120, height=15, confidence=0.95),
            
            # Chat messages
            Mock(text="Solo Tale:", x=10, y=950, width=80, height=20, confidence=0.90),
            Mock(text="[16:45] Solo Tale:", x=10, y=970, width=140, height=20, confidence=0.88),
            
            # Interface numbers
            Mock(text="99", x=800, y=120, width=25, height=20, confidence=0.95),
            Mock(text="85", x=800, y=160, width=25, height=20, confidence=0.93),
        ]
        
        with patch.object(self.detector.ocr, 'detect_text') as mock_ocr:
            mock_ocr.return_value = runelite_detections
            
            # Create a RuneLite-sized screenshot
            runelite_screenshot = np.zeros((1076, 2168, 3), dtype=np.uint8)
            
            results = self.detector.detect_all_text(runelite_screenshot)
            
            # Should detect overlays
            self.assertGreater(len(results['overlays']), 0)
            
            # Should detect chat
            self.assertGreater(len(results['chat_messages']), 0)
            
            # Should detect numbers
            self.assertGreater(len(results['numbers']), 0)


def run_tests():
    """Run all adaptive text detection tests"""
    print("🧪 Running Adaptive Text Detection Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveTextDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectedTextArea))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithRealData))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests() 