import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from bots.example_bot import ExampleCombatBot


@pytest.mark.unit
class TestExampleCombatBot:
    """Test ExampleCombatBot implementation"""
    
    def test_initialization(self, sample_config):
        """Test bot initialization"""
        bot = ExampleCombatBot(sample_config)
        
        assert bot.name == "ExampleCombatBot"
        assert bot.config == sample_config
        assert bot.target_monster == sample_config["settings"]["target_monster"]
    
    @patch('bots.example_bot.cv_system')
    @patch('bots.example_bot.screen_capture')
    def test_initialize(self, mock_screen, mock_cv, sample_config):
        """Test bot initialization method"""
        # Mock required templates exist - return a non-None value for each template
        mock_template = Mock()
        mock_cv.template_manager.get_template.return_value = mock_template
        mock_screen.is_client_active.return_value = True
        
        bot = ExampleCombatBot(sample_config)
        
        # Verify templates are checked for goblin, bread, health_bar
        result = bot.initialize()
        
        assert result is True
        # Verify that get_template was called for each required template
        assert mock_cv.template_manager.get_template.call_count >= 3
    
    @patch('bots.example_bot.cv_system')
    @patch('bots.example_bot.screen_capture')
    def test_initialize_missing_templates(self, mock_screen, mock_cv, sample_config):
        """Test initialization with missing templates"""
        # Mock missing templates
        mock_cv.template_manager.get_template.return_value = None
        mock_screen.is_client_active.return_value = True
        
        bot = ExampleCombatBot(sample_config)
        result = bot.initialize()
        
        assert result is False
    
    def test_execute_cycle(self, sample_config):
        """Test bot execute_cycle method"""
        bot = ExampleCombatBot(sample_config)
        
        # Return a truthy value for capture_and_process (empty dict is falsy)
        mock_game_state = {'timestamp': time.time(), 'detections': []}
        
        with patch.object(bot, 'capture_and_process', return_value=mock_game_state):
            with patch.object(bot, '_should_eat_food', return_value=False):
                with patch.object(bot, '_is_in_combat', return_value=False):
                    with patch.object(bot, '_find_and_attack_monster', return_value=True):
                        result = bot.execute_cycle()
                        
                        assert result is True
    
    def test_should_eat_food(self, sample_config):
        """Test food eating logic"""
        bot = ExampleCombatBot(sample_config)
        
        # Set last_food_time to recent time so it shouldn't eat immediately
        bot.last_food_time = time.time()
        
        result = bot._should_eat_food()
        assert result is False
        
        # Test that it should eat after sufficient time has passed
        bot.last_food_time = time.time() - 35.0  # 35 seconds ago
        result = bot._should_eat_food()
        assert result is True
    
    def test_combat_detection(self, sample_config):
        """Test combat detection logic"""
        bot = ExampleCombatBot(sample_config)
        
        # Not in combat initially
        assert bot._is_in_combat() is False
        
        # Set combat target
        bot.current_target = time.time()
        assert bot._is_in_combat() is True
        
        # Combat timeout
        bot.current_target = time.time() - 20  # 20 seconds ago
        assert bot._is_in_combat() is False
    
    def test_get_status(self, sample_config):
        """Test extended status reporting"""
        bot = ExampleCombatBot(sample_config)
        bot.monsters_killed = 5
        bot.food_consumed = 3
        
        status = bot.get_status()
        
        assert "monsters_killed" in status
        assert "food_consumed" in status
        assert "in_combat" in status
        assert "target_monster" in status
        assert status["monsters_killed"] == 5
        assert status["food_consumed"] == 3
    
    def test_cleanup(self, sample_config):
        """Test bot cleanup method"""
        bot = ExampleCombatBot(sample_config)
        
        # Should not raise any exceptions
        try:
            bot.cleanup()
        except Exception as e:
            pytest.fail(f"cleanup() raised {e} unexpectedly!")
    
    @patch('core.bot_base.screen_capture')
    def test_integration_with_base_class(self, mock_screen_capture, sample_config):
        """Test that ExampleCombatBot properly extends BotBase"""
        from core.bot_base import BotBase
        
        bot = ExampleCombatBot(sample_config)
        
        # Test it inherits from BotBase
        assert isinstance(bot, BotBase)
        
        # Test it has the required abstract methods implemented
        assert hasattr(bot, 'initialize')
        assert hasattr(bot, 'execute_cycle')
        assert callable(bot.initialize)
        assert callable(bot.execute_cycle)