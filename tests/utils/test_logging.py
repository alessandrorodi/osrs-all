import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from utils.logging import setup_logging, get_logger, BotLogger


@pytest.mark.unit
class TestSetupLogging:
    """Test setup_logging function"""
    
    @patch('logging.getLogger')
    @patch('logging.handlers.RotatingFileHandler')
    @patch('logging.StreamHandler')
    def test_setup_logging_default(self, mock_stream, mock_file, mock_get_logger):
        """Test default logging setup"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        result = setup_logging()
        
        assert result == mock_logger
        mock_get_logger.assert_called_with("osrs_bot")
        mock_logger.setLevel.assert_called()
    
    @patch('logging.getLogger')
    def test_setup_logging_custom_name(self, mock_get_logger):
        """Test logging setup with custom name"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        setup_logging("custom_bot")
        
        mock_get_logger.assert_called_with("custom_bot")
    
    @patch('logging.getLogger')
    def test_setup_logging_custom_level(self, mock_get_logger):
        """Test logging setup with custom level"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        setup_logging(level="ERROR")
        
        mock_logger.setLevel.assert_called_with(logging.ERROR)


@pytest.mark.unit
class TestGetLogger:
    """Test get_logger function"""
    
    @patch('utils.logging.setup_logging')
    def test_get_logger(self, mock_setup):
        """Test get_logger function"""
        mock_logger = Mock()
        mock_setup.return_value = mock_logger
        
        result = get_logger("test_logger")
        
        assert result == mock_logger
        mock_setup.assert_called_with("test_logger")


@pytest.mark.unit
class TestBotLogger:
    """Test BotLogger class"""
    
    @patch('utils.logging.get_logger')
    def test_initialization(self, mock_get_logger):
        """Test BotLogger initialization"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        bot_logger = BotLogger("test_bot")
        
        assert bot_logger.bot_name == "test_bot"
        assert bot_logger.logger == mock_logger
        assert bot_logger.action_count == 0
        assert bot_logger.error_count == 0
        mock_get_logger.assert_called_with("bot.test_bot")
    
    @patch('utils.logging.get_logger')
    def test_log_action(self, mock_get_logger):
        """Test action logging"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        bot_logger = BotLogger("test_bot")
        bot_logger.log_action("click_button", "details")
        
        assert bot_logger.action_count == 1
        mock_logger.info.assert_called_once()
        
        # Check the logged message format
        call_args = mock_logger.info.call_args[0][0]
        assert "[Action #1]" in call_args
        assert "click_button" in call_args
        assert "details" in call_args
    
    @patch('utils.logging.get_logger')
    def test_log_detection(self, mock_get_logger):
        """Test detection logging"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        bot_logger = BotLogger("test_bot")
        
        # Test with detections found
        bot_logger.log_detection("health_bars", 2, 0.85)
        mock_logger.debug.assert_called()
        
        # Test with no detections
        bot_logger.log_detection("enemies", 0)
        assert mock_logger.debug.call_count == 2
    
    @patch('utils.logging.get_logger')
    def test_log_error(self, mock_get_logger):
        """Test error logging"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        bot_logger = BotLogger("test_bot")
        test_exception = Exception("Test error")
        
        bot_logger.log_error("Something went wrong", test_exception)
        
        assert bot_logger.error_count == 1
        mock_logger.error.assert_called_once()
        
        # Check exc_info parameter was passed
        call_kwargs = mock_logger.error.call_args[1]
        assert call_kwargs.get('exc_info') == test_exception
    
    @patch('utils.logging.get_logger')
    def test_log_state_change(self, mock_get_logger):
        """Test state change logging"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        bot_logger = BotLogger("test_bot")
        bot_logger.log_state_change("idle", "running")
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "State change: idle -> running" in call_args
    
    @patch('utils.logging.get_logger')
    def test_log_performance(self, mock_get_logger):
        """Test performance logging"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        bot_logger = BotLogger("test_bot")
        stats = {"actions_per_minute": 10.5, "errors": 2}
        
        bot_logger.log_performance(stats)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "Performance:" in call_args
    
    @patch('utils.logging.get_logger')
    def test_get_stats(self, mock_get_logger):
        """Test stats retrieval"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        bot_logger = BotLogger("test_bot")
        bot_logger.action_count = 10
        bot_logger.error_count = 2
        
        stats = bot_logger.get_stats()
        
        assert stats["actions"] == 10
        assert stats["errors"] == 2
        assert stats["error_rate"] == 0.2  # 2/10