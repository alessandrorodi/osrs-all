"""
Logging utilities for OSRS Bot Framework
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
import colorlog

from config.settings import LOGGING, LOGS_DIR


def setup_logging(name: Optional[str] = None, level: Optional[str] = None,
                 file_logging: bool = True, console_logging: bool = True) -> logging.Logger:
    """
    Set up logging with file and console handlers
    
    Args:
        name: Logger name (defaults to root logger)
        level: Logging level override
        file_logging: Enable file logging
        console_logging: Enable console logging
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger_name = name or "osrs_bot"
    logger = logging.getLogger(logger_name)
    
    # Set level
    log_level = level or LOGGING["level"]
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(LOGGING["format"])
    
    # Colored console formatter
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    # File handler with rotation
    if file_logging and LOGGING["file_logging"]:
        log_file = Path(LOGS_DIR) / f"{logger_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOGGING["max_log_size"],
            backupCount=LOGGING["backup_count"],
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console_logging and LOGGING["console_logging"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with the standard configuration"""
    return setup_logging(name)


class BotLogger:
    """Specialized logger for bot operations"""
    
    def __init__(self, bot_name: str):
        self.bot_name = bot_name
        self.logger = get_logger(f"bot.{bot_name}")
        self.action_count = 0
        self.error_count = 0
    
    def log_action(self, action: str, details: Optional[str] = None) -> None:
        """Log a bot action"""
        self.action_count += 1
        message = f"[Action #{self.action_count}] {action}"
        if details:
            message += f" - {details}"
        self.logger.info(message)
    
    def log_detection(self, detection_type: str, count: int, confidence: float = 0.0) -> None:
        """Log a computer vision detection"""
        if count > 0:
            message = f"Detected {count} {detection_type}"
            if confidence > 0:
                message += f" (confidence: {confidence:.2f})"
            self.logger.debug(message)
        else:
            self.logger.debug(f"No {detection_type} detected")
    
    def log_error(self, error: str, exception: Optional[Exception] = None) -> None:
        """Log a bot error"""
        self.error_count += 1
        message = f"[Error #{self.error_count}] {error}"
        if exception:
            self.logger.error(message, exc_info=exception)
        else:
            self.logger.error(message)
    
    def log_state_change(self, old_state: str, new_state: str) -> None:
        """Log a bot state change"""
        self.logger.info(f"State change: {old_state} -> {new_state}")
    
    def log_performance(self, stats: dict) -> None:
        """Log performance statistics"""
        self.logger.info(f"Performance: {stats}")
    
    def get_stats(self) -> dict:
        """Get logging statistics"""
        return {
            "actions": self.action_count,
            "errors": self.error_count,
            "error_rate": self.error_count / max(1, self.action_count)
        }


# Initialize root logger
root_logger = setup_logging("osrs_bot_framework") 