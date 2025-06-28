import pytest
import numpy as np
import cv2
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
import tempfile
import threading
import time


@pytest.fixture
def mock_screen_image():
    """Create a mock screen image for testing"""
    return np.zeros((600, 800, 3), dtype=np.uint8)


@pytest.fixture
def mock_template_image():
    """Create a mock template image for testing"""
    return np.ones((50, 50, 3), dtype=np.uint8) * 128


@pytest.fixture
def mock_client_region():
    """Mock client region for testing"""
    from types import SimpleNamespace
    return SimpleNamespace(x=100, y=100, width=765, height=503)


@pytest.fixture
def mock_pyautogui():
    """Mock pyautogui for automation tests"""
    with patch('pyautogui.position') as mock_pos, \
         patch('pyautogui.moveTo') as mock_move, \
         patch('pyautogui.click') as mock_click, \
         patch('pyautogui.write') as mock_write, \
         patch('pyautogui.press') as mock_press, \
         patch('pyautogui.hotkey') as mock_hotkey, \
         patch('pyautogui.scroll') as mock_scroll, \
         patch('pyautogui.drag') as mock_drag:
        
        mock_pos.return_value = (400, 300)
        mock_move.return_value = None
        mock_click.return_value = None
        mock_write.return_value = None
        mock_press.return_value = None
        mock_hotkey.return_value = None
        mock_scroll.return_value = None
        mock_drag.return_value = None
        
        yield {
            'position': mock_pos,
            'moveTo': mock_move,
            'click': mock_click,
            'write': mock_write,
            'press': mock_press,
            'hotkey': mock_hotkey,
            'scroll': mock_scroll,
            'drag': mock_drag
        }


@pytest.fixture
def mock_cv2():
    """Mock cv2 for computer vision tests"""
    with patch('cv2.imread') as mock_imread, \
         patch('cv2.imwrite') as mock_imwrite, \
         patch('cv2.matchTemplate') as mock_match, \
         patch('cv2.cvtColor') as mock_color, \
         patch('cv2.findContours') as mock_contours, \
         patch('cv2.boundingRect') as mock_rect, \
         patch('cv2.rectangle') as mock_draw_rect, \
         patch('cv2.putText') as mock_text:
        
        # Mock imread to return test image
        mock_imread.return_value = np.ones((50, 50, 3), dtype=np.uint8) * 128
        mock_imwrite.return_value = True
        
        # Mock template matching
        match_result = np.ones((10, 10), dtype=np.float32) * 0.9
        mock_match.return_value = match_result
        
        # Mock color conversion
        mock_color.return_value = np.ones((50, 50, 3), dtype=np.uint8)
        
        # Mock contour detection
        contour = np.array([[[10, 10]], [[60, 10]], [[60, 60]], [[10, 60]]])
        mock_contours.return_value = ([contour], None)
        mock_rect.return_value = (10, 10, 50, 50)
        
        yield {
            'imread': mock_imread,
            'imwrite': mock_imwrite,
            'matchTemplate': mock_match,
            'cvtColor': mock_color,
            'findContours': mock_contours,
            'boundingRect': mock_rect,
            'rectangle': mock_draw_rect,
            'putText': mock_text
        }


@pytest.fixture
def mock_mss():
    """Mock mss for screen capture tests"""
    with patch('mss.mss') as mock_mss_class:
        mock_sct = Mock()
        mock_sct.grab.return_value = Mock(
            rgb=b'\x00' * (800 * 600 * 3),
            size=(800, 600)
        )
        mock_mss_class.return_value = mock_sct
        yield mock_sct


@pytest.fixture
def temp_dir():
    """Create temporary directory for file tests"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_logging():
    """Mock logging for tests"""
    with patch('logging.getLogger') as mock_get_logger:
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.debug = Mock()
        mock_logger.warning = Mock()
        mock_logger.error = Mock()
        mock_logger.critical = Mock()
        mock_get_logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture
def sample_config():
    """Sample bot configuration for testing"""
    return {
        "name": "test_bot",
        "enabled": True,
        "settings": {
            "target_monster": "goblin",
            "combat_style": "melee",
            "food_threshold": 50
        },
        "templates": ["goblin", "health_bar", "food_item"],
        "regions": {
            "inventory": [548, 205, 190, 261],
            "minimap": [570, 9, 146, 151],
            "chat": [7, 345, 506, 120]
        }
    }


@pytest.fixture
def mock_threading():
    """Mock threading for bot tests"""
    with patch('threading.Thread') as mock_thread_class, \
         patch('threading.Event') as mock_event_class:
        
        # Mock Thread
        mock_thread = Mock()
        mock_thread.start = Mock()
        mock_thread.join = Mock()
        mock_thread.is_alive.return_value = False
        mock_thread_class.return_value = mock_thread
        
        # Mock Event
        mock_event = Mock()
        mock_event.is_set.return_value = False
        mock_event.set = Mock()
        mock_event.clear = Mock()
        mock_event_class.return_value = mock_event
        
        yield {
            'Thread': mock_thread_class,
            'Event': mock_event_class,
            'thread_instance': mock_thread,
            'event_instance': mock_event
        }


@pytest.fixture
def mock_time():
    """Mock time module for deterministic testing"""
    with patch('time.time') as mock_time_func, \
         patch('time.sleep') as mock_sleep:
        
        # Start time counter
        start_time = 1000.0
        mock_time_func.side_effect = lambda: start_time + len(mock_time_func.call_args_list)
        mock_sleep.return_value = None
        
        yield {
            'time': mock_time_func,
            'sleep': mock_sleep
        }