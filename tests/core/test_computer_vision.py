import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from core.computer_vision import (
    Detection, TemplateManager, TemplateMatching, ColorDetection, 
    FeatureDetection, ComputerVision, cv_system
)


@pytest.mark.unit
class TestDetection:
    """Test Detection dataclass"""
    
    def test_detection_creation(self):
        """Test detection object creation"""
        det = Detection(x=10, y=20, width=50, height=30, confidence=0.8, label="test")
        
        assert det.x == 10
        assert det.y == 20
        assert det.width == 50
        assert det.height == 30
        assert det.confidence == 0.8
        assert det.label == "test"
    
    def test_detection_center(self):
        """Test center property calculation"""
        det = Detection(x=10, y=20, width=50, height=30, confidence=0.8)
        assert det.center == (35, 35)  # (10+25, 20+15)
    
    def test_detection_bbox(self):
        """Test bounding box property"""
        det = Detection(x=10, y=20, width=50, height=30, confidence=0.8)
        assert det.bbox == (10, 20, 50, 30)


@pytest.mark.unit
class TestTemplateManager:
    """Test TemplateManager class"""
    
    @patch('cv2.imread')
    @patch('pathlib.Path.glob')
    @patch('pathlib.Path.exists')
    def test_load_templates(self, mock_exists, mock_glob, mock_imread):
        """Test template loading"""
        mock_exists.return_value = True
        mock_template_file = Mock()
        mock_template_file.stem = "test_template"
        mock_glob.return_value = [mock_template_file]
        mock_imread.return_value = np.ones((50, 50, 3), dtype=np.uint8)
        
        manager = TemplateManager()
        
        assert "test_template" in manager.templates
        assert manager.template_sizes["test_template"] == (50, 50)
    
    def test_add_template(self):
        """Test adding template"""
        manager = TemplateManager()
        test_image = np.ones((30, 40, 3), dtype=np.uint8)
        
        manager.add_template("new_template", test_image)
        
        assert "new_template" in manager.templates
        assert manager.template_sizes["new_template"] == (40, 30)
    
    def test_get_template(self):
        """Test getting template"""
        manager = TemplateManager()
        test_image = np.ones((30, 40, 3), dtype=np.uint8)
        manager.add_template("test", test_image)
        
        retrieved = manager.get_template("test")
        assert np.array_equal(retrieved, test_image)
        
        # Test non-existent template
        assert manager.get_template("nonexistent") is None


@pytest.mark.unit
class TestTemplateMatching:
    """Test TemplateMatching class"""
    
    def test_initialization(self):
        """Test template matching initialization"""
        manager = Mock()
        matcher = TemplateMatching(manager)
        
        assert matcher.template_manager == manager
        assert hasattr(matcher, 'threshold')
        assert hasattr(matcher, 'method')
        assert hasattr(matcher, 'max_matches')
    
    @patch('cv2.matchTemplate')
    @patch('numpy.where')
    def test_match_template(self, mock_where, mock_match_template):
        """Test template matching functionality"""
        manager = Mock()
        matcher = TemplateMatching(manager)
        
        # Mock cv2.matchTemplate result
        result = np.ones((10, 10)) * 0.9
        mock_match_template.return_value = result
        
        # Mock numpy.where result for high confidence matches
        mock_where.return_value = ([5], [5])  # One match at (5,5)
        
        image = np.ones((100, 100, 3), dtype=np.uint8)
        template = np.ones((20, 20, 3), dtype=np.uint8)
        
        detections = matcher.match_template(image, template, "test_label")
        
        assert len(detections) == 1
        assert detections[0].x == 5
        assert detections[0].y == 5
        assert detections[0].label == "test_label"
    
    def test_iou_calculation(self):
        """Test IoU calculation"""
        manager = Mock()
        matcher = TemplateMatching(manager)
        
        det1 = Detection(x=0, y=0, width=10, height=10, confidence=0.8)
        det2 = Detection(x=5, y=5, width=10, height=10, confidence=0.9)
        
        iou = matcher._iou(det1, det2)
        # Expected IoU: intersection(25) / union(175) = 0.142857
        assert 0.14 < iou < 0.15


@pytest.mark.unit
class TestColorDetection:
    """Test ColorDetection class"""
    
    @patch('cv2.cvtColor')
    @patch('cv2.inRange')
    @patch('cv2.findContours')
    @patch('cv2.contourArea')
    @patch('cv2.boundingRect')
    def test_find_color(self, mock_rect, mock_area, mock_contours, mock_range, mock_color):
        """Test color detection"""
        detector = ColorDetection()
        
        # Mock cv2 functions
        mock_color.return_value = np.ones((100, 100, 3), dtype=np.uint8)
        mock_range.return_value = np.ones((100, 100), dtype=np.uint8)
        
        # Mock contour
        contour = np.array([[[10, 10]], [[60, 10]], [[60, 60]], [[10, 60]]])
        mock_contours.return_value = ([contour], None)
        mock_area.return_value = 2500  # Large enough area
        mock_rect.return_value = (10, 10, 50, 50)
        
        image = np.ones((100, 100, 3), dtype=np.uint8)
        detections = detector.find_color(image, (120, 255, 255))  # Red in HSV
        
        assert len(detections) == 1
        assert detections[0].x == 10
        assert detections[0].y == 10


@pytest.mark.unit
class TestComputerVision:
    """Test main ComputerVision class"""
    
    def test_initialization(self):
        """Test computer vision system initialization"""
        cv = ComputerVision()
        
        assert hasattr(cv, 'template_manager')
        assert hasattr(cv, 'template_matching')
        assert hasattr(cv, 'color_detection')
        assert hasattr(cv, 'feature_detection')
    
    @patch('cv2.getTickCount')
    def test_process_image(self, mock_tick):
        """Test image processing"""
        mock_tick.return_value = 1000
        
        cv = ComputerVision()
        image = np.ones((100, 100, 3), dtype=np.uint8)
        
        with patch.object(cv.color_detection, 'find_health_bar', return_value=None):
            with patch.object(cv.feature_detection, 'extract_features', return_value=([], None)):
                result = cv.process_image(image)
        
        assert "timestamp" in result
        assert "image_shape" in result
        assert "detections" in result
        assert "health_bars" in result
        assert "features" in result