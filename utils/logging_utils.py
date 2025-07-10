"""
Logging utilities and colored console output.
"""

import logging
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(log_level: str = "INFO", log_file: str = "bathymetric_processing.log"):
    """Enhanced logging setup with colored output."""
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.FileHandler(f"logs/{log_file}"),
            logging.StreamHandler()
        ]
    )
    
    console_handler = logging.getLogger().handlers[1]
    console_handler.setFormatter(ColoredFormatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    ))
