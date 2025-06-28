import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from core.automation import HumanMouse, HumanKeyboard, MouseSpeed, random_delay, emergency_stop


@pytest.mark.unit
class TestHumanMouse:
    """Test HumanMouse class"""
    
    @patch('pyautogui.position')
    def test_initialization(self, mock_position):
        """Test mouse initialization"""
        mock_position.return_value = (100, 200)
        mouse = HumanMouse()
        
        assert mouse.last_position == (100, 200)
        assert mouse.movement_history == []
    
    def test_distance_calculation(self):
        """Test distance calculation between points"""
        mouse = HumanMouse()
        distance = mouse._distance((0, 0), (3, 4))
        assert distance == 5.0  # 3-4-5 triangle
        
        distance = mouse._distance((10, 10), (10, 10))
        assert distance == 0.0
    
    def test_add_noise(self):
        """Test noise addition to coordinates"""
        mouse = HumanMouse()
        
        # Test with randomization enabled
        with patch('core.automation.AUTOMATION', {'mouse': {'randomization': True}}):
            with patch('random.randint', return_value=1):
                noisy_point = mouse._add_noise((100, 100), intensity=1.0)
                assert noisy_point == (101, 101)
        
        # Test with randomization disabled
        with patch('core.automation.AUTOMATION', {'mouse': {'randomization': False}}):
            noisy_point = mouse._add_noise((100, 100), intensity=1.0)
            assert noisy_point == (100, 100)
    
    def test_bezier_curve_generation(self):
        """Test bezier curve point generation"""
        mouse = HumanMouse()
        
        with patch('random.randint', return_value=0):
            points = mouse._bezier_curve((0, 0), (100, 100))
            
            assert len(points) > 10
            assert points[0] == (0, 0)
            assert points[-1] == (100, 100)
    
    @patch('pyautogui.position')
    @patch('pyautogui.moveTo')
    @patch('time.sleep')
    def test_move_to_instant(self, mock_sleep, mock_move, mock_position):
        """Test instant mouse movement"""
        mock_position.return_value = (0, 0)
        
        # Disable randomization for this test
        with patch('core.automation.AUTOMATION', {
            'mouse': {'randomization': False, 'smooth_movement': False}
        }):
            mouse = HumanMouse()
            result = mouse.move_to(100, 100, speed="instant", smooth=False)
            
            assert result is True
            mock_move.assert_called_once_with(100, 100)
            mock_sleep.assert_not_called()
    
    @patch('pyautogui.position')
    @patch('pyautogui.moveTo')
    @patch('time.sleep')
    def test_move_to_smooth(self, mock_sleep, mock_move, mock_position):
        """Test smooth mouse movement"""
        mock_position.return_value = (0, 0)
        
        with patch('core.automation.AUTOMATION', {
            'mouse': {'smooth_movement': True, 'randomization': False}
        }):
            mouse = HumanMouse()
            result = mouse.move_to(100, 100, speed="human", smooth=True)
            
            assert result is True
            assert mock_move.call_count > 1  # Multiple movement calls for smooth movement
    
    @patch('pyautogui.position')
    def test_move_to_same_position(self, mock_position):
        """Test movement to same position (should skip)"""
        mock_position.return_value = (100, 100)
        
        mouse = HumanMouse()
        result = mouse.move_to(102, 102)  # Within tolerance
        
        assert result is True
    
    @patch('pyautogui.position')
    @patch('pyautogui.click')
    @patch('time.sleep')
    def test_click_at_position(self, mock_sleep, mock_click, mock_position):
        """Test clicking at specific position"""
        mock_position.return_value = (50, 50)
        
        with patch.object(HumanMouse, 'move_to', return_value=True) as mock_move:
            mouse = HumanMouse()
            result = mouse.click(100, 100, button="left", clicks=1)
            
            assert result is True
            mock_move.assert_called_once_with(100, 100)
            mock_click.assert_called_once_with(button="left")
    
    @patch('pyautogui.position')
    @patch('pyautogui.click')
    @patch('time.sleep')
    def test_click_multiple(self, mock_sleep, mock_click, mock_position):
        """Test multiple clicks"""
        mock_position.return_value = (100, 100)
        
        mouse = HumanMouse()
        result = mouse.click(clicks=3, interval=0.1)
        
        assert result is True
        assert mock_click.call_count == 3
    
    @patch('pyautogui.drag')
    def test_drag(self, mock_drag):
        """Test mouse drag operation"""
        with patch.object(HumanMouse, 'move_to', return_value=True) as mock_move:
            mouse = HumanMouse()
            result = mouse.drag(0, 0, 100, 100, duration=0.5)
            
            assert result is True
            mock_move.assert_called_once_with(0, 0)
            mock_drag.assert_called_once_with(100, 100, duration=0.5, button="left")
    
    @patch('pyautogui.scroll')
    def test_scroll(self, mock_scroll):
        """Test mouse scroll operation"""
        with patch.object(HumanMouse, 'move_to', return_value=True) as mock_move:
            mouse = HumanMouse()
            result = mouse.scroll(5, x=100, y=100)
            
            assert result is True
            mock_move.assert_called_once_with(100, 100)
            mock_scroll.assert_called_once_with(5)
    
    @patch('pyautogui.position')
    @patch('pyautogui.moveTo')
    def test_move_to_error_handling(self, mock_move, mock_position):
        """Test error handling in move_to"""
        mock_position.return_value = (0, 0)
        mock_move.side_effect = Exception("Test error")
        
        mouse = HumanMouse()
        result = mouse.move_to(100, 100)
        
        assert result is False


@pytest.mark.unit
class TestHumanKeyboard:
    """Test HumanKeyboard class"""
    
    def test_initialization(self):
        """Test keyboard initialization"""
        with patch('core.automation.AUTOMATION', {
            'keyboard': {'typing_speed': 0.05, 'randomization': True}
        }):
            keyboard = HumanKeyboard()
            
            assert keyboard.typing_speed == 0.05
            assert keyboard.randomization is True
    
    @patch('pyautogui.write')
    @patch('time.sleep')
    def test_type_text_without_randomization(self, mock_sleep, mock_write):
        """Test typing text without randomization"""
        keyboard = HumanKeyboard()
        keyboard.randomization = False
        keyboard.typing_speed = 0.1
        
        result = keyboard.type_text("hello")
        
        assert result is True
        assert mock_write.call_count == 5  # One call per character
        assert mock_sleep.call_count == 5
    
    @patch('pyautogui.write')
    @patch('time.sleep')
    @patch('random.uniform')
    def test_type_text_with_randomization(self, mock_random, mock_sleep, mock_write):
        """Test typing text with randomization"""
        mock_random.return_value = 0.05
        
        keyboard = HumanKeyboard()
        keyboard.randomization = True
        keyboard.typing_speed = 0.1
        
        result = keyboard.type_text("hi")
        
        assert result is True
        assert mock_write.call_count == 2
        assert mock_sleep.call_count == 2
    
    @patch('pyautogui.press')
    @patch('time.sleep')
    def test_press_key_single(self, mock_sleep, mock_press):
        """Test pressing a single key"""
        keyboard = HumanKeyboard()
        keyboard.randomization = False
        
        result = keyboard.press_key("enter")
        
        assert result is True
        mock_press.assert_called_once_with("enter")
    
    @patch('pyautogui.press')
    @patch('time.sleep')
    def test_press_key_multiple(self, mock_sleep, mock_press):
        """Test pressing key multiple times"""
        keyboard = HumanKeyboard()
        keyboard.randomization = False
        
        result = keyboard.press_key("space", presses=3, interval=0.1)
        
        assert result is True
        assert mock_press.call_count == 3
    
    @patch('pyautogui.hotkey')
    def test_key_combination(self, mock_hotkey):
        """Test key combination"""
        keyboard = HumanKeyboard()
        
        result = keyboard.key_combination(['ctrl', 'c'])
        
        assert result is True
        mock_hotkey.assert_called_once_with('ctrl', 'c')
    
    @patch('pyautogui.write')
    def test_type_text_error_handling(self, mock_write):
        """Test error handling in type_text"""
        mock_write.side_effect = Exception("Test error")
        
        keyboard = HumanKeyboard()
        result = keyboard.type_text("test")
        
        assert result is False


@pytest.mark.unit
class TestMouseSpeed:
    """Test MouseSpeed enum"""
    
    def test_mouse_speed_values(self):
        """Test mouse speed enum values"""
        assert MouseSpeed.INSTANT.value == 0
        assert MouseSpeed.FAST.value == 0.1
        assert MouseSpeed.HUMAN.value == 0.3
        assert MouseSpeed.SLOW.value == 0.5


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility functions"""
    
    @patch('time.sleep')
    @patch('random.uniform')
    def test_random_delay_default(self, mock_uniform, mock_sleep):
        """Test random delay with default parameters"""
        mock_uniform.return_value = 0.2
        
        with patch('core.automation.AUTOMATION', {
            'delays': {'min_action_delay': 0.1, 'max_action_delay': 0.3}
        }):
            random_delay()
            
            mock_uniform.assert_called_once_with(0.1, 0.3)
            mock_sleep.assert_called_once_with(0.2)
    
    @patch('time.sleep')
    @patch('random.uniform')
    def test_random_delay_custom(self, mock_uniform, mock_sleep):
        """Test random delay with custom parameters"""
        mock_uniform.return_value = 0.5
        
        random_delay(min_delay=0.2, max_delay=0.8)
        
        mock_uniform.assert_called_once_with(0.2, 0.8)
        mock_sleep.assert_called_once_with(0.5)
    
    @patch('pyautogui.moveTo')
    def test_emergency_stop(self, mock_move):
        """Test emergency stop function"""
        with patch('core.automation.SAFETY', {
            'failsafe': {'mouse_corner': True}
        }):
            with pytest.raises(KeyboardInterrupt):
                emergency_stop()
            
            mock_move.assert_called_once_with(0, 0)


@pytest.mark.unit
class TestGlobalInstances:
    """Test global mouse and keyboard instances"""
    
    def test_global_mouse_instance(self):
        """Test that global mouse instance exists"""
        from core.automation import mouse
        assert isinstance(mouse, HumanMouse)
    
    def test_global_keyboard_instance(self):
        """Test that global keyboard instance exists"""
        from core.automation import keyboard
        assert isinstance(keyboard, HumanKeyboard)