import logging
import os
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        return super().format(record)

def setup_logging(log_level="INFO"):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/bathymetric_processing.log"),
            logging.StreamHandler()
        ]
    )
