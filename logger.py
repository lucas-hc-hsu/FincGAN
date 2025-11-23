"""
Professional logging utility for FincGAN project.
Provides consistent logging across all modules with timestamps and proper formatting.
"""

import logging
import sys
from datetime import datetime
import pytz


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to level name
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(name='FincGAN', level=logging.INFO, use_color=True, log_to_file=True, log_dir='logs'):
    """
    Setup and return a logger with consistent formatting.

    Args:
        name (str): Logger name
        level (int): Logging level (logging.DEBUG, logging.INFO, etc.)
        use_color (bool): Whether to use colored output
        log_to_file (bool): Whether to save logs to file
        log_dir (str): Directory to save log files (default: 'logs')

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create formatter with timezone support
    timezone = pytz.timezone('Asia/Taipei')

    class TaipeiFormatter(ColoredFormatter if use_color else logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=timezone)
            if datefmt:
                return dt.strftime(datefmt)
            return dt.strftime("%m/%d/%Y, %H:%M:%S")

    # Format: [timestamp] message
    formatter = TaipeiFormatter('[%(asctime)s] %(message)s', datefmt='%m/%d/%Y, %H:%M:%S')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # Add file handler if requested
    if log_to_file:
        import os
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create log filename with timestamp
        timestamp = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f'fincgan_{timestamp}.log')

        # Create file handler
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setLevel(level)

        # Use plain formatter for file (no colors)
        file_formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%m/%d/%Y, %H:%M:%S')

        class TaipeiFileFormatter(logging.Formatter):
            def formatTime(self, record, datefmt=None):
                dt = datetime.fromtimestamp(record.created, tz=timezone)
                if datefmt:
                    return dt.strftime(datefmt)
                return dt.strftime("%m/%d/%Y, %H:%M:%S")

        file_formatter = TaipeiFileFormatter('[%(asctime)s] %(message)s', datefmt='%m/%d/%Y, %H:%M:%S')
        file_handler.setFormatter(file_formatter)

        logger.addHandler(file_handler)

        # Log the file location
        logger.info(f"Logging to file: {log_filename}")

    return logger


def get_logger(name='FincGAN', level=logging.INFO):
    """
    Get or create a logger instance.

    Args:
        name (str): Logger name
        level (int): Logging level

    Returns:
        logging.Logger: Logger instance
    """
    return setup_logger(name, level)


# Default logger instance for quick import
default_logger = get_logger()
