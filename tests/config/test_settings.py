import pytest
from unittest.mock import patch
from pathlib import Path

# Import settings module
import config.settings as settings


@pytest.mark.unit
class TestSettings:
    """Test configuration settings"""
    
    def test_project_paths_exist(self):
        """Test that all project path constants are defined"""
        assert hasattr(settings, 'PROJECT_ROOT')
        assert hasattr(settings, 'DATA_DIR')
        assert hasattr(settings, 'TEMPLATES_DIR')
        assert hasattr(settings, 'MODELS_DIR')
        assert hasattr(settings, 'LOGS_DIR')
        assert hasattr(settings, 'CONFIG_DIR')
        
        # Check they are Path objects
        assert isinstance(settings.PROJECT_ROOT, Path)
        assert isinstance(settings.DATA_DIR, Path)
        assert isinstance(settings.TEMPLATES_DIR, Path)
        assert isinstance(settings.MODELS_DIR, Path)
        assert isinstance(settings.LOGS_DIR, Path)
        assert isinstance(settings.CONFIG_DIR, Path)
    
    def test_screen_capture_settings(self):
        """Test screen capture configuration"""
        assert 'fps' in settings.SCREEN_CAPTURE
        assert 'region' in settings.SCREEN_CAPTURE
        assert 'compression' in settings.SCREEN_CAPTURE
        
        assert isinstance(settings.SCREEN_CAPTURE['fps'], int)
        assert settings.SCREEN_CAPTURE['fps'] > 0
    
    def test_vision_settings(self):
        """Test computer vision configuration"""
        assert 'template_matching' in settings.VISION
        assert 'feature_detection' in settings.VISION
        assert 'color_detection' in settings.VISION
        
        # Template matching settings
        tm = settings.VISION['template_matching']
        assert 'threshold' in tm
        assert 'method' in tm
        assert 'max_matches' in tm
        assert 0 < tm['threshold'] <= 1
        assert tm['max_matches'] > 0
        
        # Feature detection settings  
        fd = settings.VISION['feature_detection']
        assert 'algorithm' in fd
        assert 'max_features' in fd
        assert 'match_threshold' in fd
        assert fd['algorithm'] in ['ORB', 'SIFT', 'SURF']
        
        # Color detection settings
        cd = settings.VISION['color_detection']
        assert 'hsv_tolerance' in cd
        assert 'min_area' in cd
        assert cd['hsv_tolerance'] > 0
        assert cd['min_area'] > 0
    
    def test_automation_settings(self):
        """Test automation configuration"""
        assert 'mouse' in settings.AUTOMATION
        assert 'keyboard' in settings.AUTOMATION
        assert 'delays' in settings.AUTOMATION
        
        # Mouse settings
        mouse = settings.AUTOMATION['mouse']
        assert 'speed' in mouse
        assert 'randomization' in mouse
        assert 'smooth_movement' in mouse
        assert mouse['speed'] in ['instant', 'fast', 'human', 'slow']
        assert isinstance(mouse['randomization'], bool)
        assert isinstance(mouse['smooth_movement'], bool)
        
        # Keyboard settings
        keyboard = settings.AUTOMATION['keyboard']
        assert 'typing_speed' in keyboard
        assert 'randomization' in keyboard
        assert keyboard['typing_speed'] > 0
        assert isinstance(keyboard['randomization'], bool)
        
        # Delay settings
        delays = settings.AUTOMATION['delays']
        assert 'min_action_delay' in delays
        assert 'max_action_delay' in delays
        assert 'anti_ban_breaks' in delays
        assert delays['min_action_delay'] >= 0
        assert delays['max_action_delay'] >= delays['min_action_delay']
        assert isinstance(delays['anti_ban_breaks'], bool)
    
    def test_safety_settings(self):
        """Test safety configuration"""
        assert 'failsafe' in settings.SAFETY
        assert 'anti_detection' in settings.SAFETY
        assert 'monitoring' in settings.SAFETY
        
        # Failsafe settings
        failsafe = settings.SAFETY['failsafe']
        assert 'enabled' in failsafe
        assert 'key' in failsafe
        assert 'mouse_corner' in failsafe
        assert isinstance(failsafe['enabled'], bool)
        assert isinstance(failsafe['mouse_corner'], bool)
        
        # Anti-detection settings
        anti_det = settings.SAFETY['anti_detection']
        assert 'randomize_timing' in anti_det
        assert 'human_patterns' in anti_det
        assert 'break_intervals' in anti_det
        assert 'long_break_chance' in anti_det
        assert isinstance(anti_det['randomize_timing'], bool)
        assert isinstance(anti_det['human_patterns'], bool)
        assert len(anti_det['break_intervals']) == 2
        assert 0 <= anti_det['long_break_chance'] <= 1
        
        # Monitoring settings
        monitoring = settings.SAFETY['monitoring']
        assert 'log_actions' in monitoring
        assert 'screenshot_errors' in monitoring
        assert 'performance_tracking' in monitoring
        assert isinstance(monitoring['log_actions'], bool)
        assert isinstance(monitoring['screenshot_errors'], bool)
        assert isinstance(monitoring['performance_tracking'], bool)
    
    def test_logging_settings(self):
        """Test logging configuration"""
        assert 'level' in settings.LOGGING
        assert 'format' in settings.LOGGING
        assert 'file_logging' in settings.LOGGING
        assert 'console_logging' in settings.LOGGING
        assert 'max_log_size' in settings.LOGGING
        assert 'backup_count' in settings.LOGGING
        
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        assert settings.LOGGING['level'] in valid_levels
        assert isinstance(settings.LOGGING['format'], str)
        assert isinstance(settings.LOGGING['file_logging'], bool)
        assert isinstance(settings.LOGGING['console_logging'], bool)
        assert settings.LOGGING['max_log_size'] > 0
        assert settings.LOGGING['backup_count'] >= 0
    
    def test_client_detection_settings(self):
        """Test client detection configuration"""
        assert 'window_title' in settings.CLIENT_DETECTION
        assert 'process_name' in settings.CLIENT_DETECTION
        assert 'auto_focus' in settings.CLIENT_DETECTION
        assert 'require_focus' in settings.CLIENT_DETECTION
        assert 'client_size' in settings.CLIENT_DETECTION
        assert 'size_tolerance' in settings.CLIENT_DETECTION
        
        assert isinstance(settings.CLIENT_DETECTION['window_title'], str)
        assert isinstance(settings.CLIENT_DETECTION['process_name'], str)
        assert isinstance(settings.CLIENT_DETECTION['auto_focus'], bool)
        assert isinstance(settings.CLIENT_DETECTION['require_focus'], bool)
        assert len(settings.CLIENT_DETECTION['client_size']) == 2
        assert settings.CLIENT_DETECTION['size_tolerance'] > 0
    
    def test_development_settings(self):
        """Test development configuration"""
        assert 'debug_mode' in settings.DEVELOPMENT
        assert 'visual_debugging' in settings.DEVELOPMENT
        assert 'save_debug_images' in settings.DEVELOPMENT
        assert 'performance_profiling' in settings.DEVELOPMENT
        
        assert isinstance(settings.DEVELOPMENT['debug_mode'], bool)
        assert isinstance(settings.DEVELOPMENT['visual_debugging'], bool)
        assert isinstance(settings.DEVELOPMENT['save_debug_images'], bool)
        assert isinstance(settings.DEVELOPMENT['performance_profiling'], bool)