import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler




def setup_logging():
    """Setup centralized logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # TimedRotatingFileHandler: rotate daily at midnight, keep 7 backups
    file_handler = TimedRotatingFileHandler(
        filename="logs/bot.log",
        when="midnight",  # rotate every midnight
        interval=1,  # every 1 day
        backupCount=7,  # keep last 7 log files
        encoding="utf-8"
    )

    # Add suffix to rotated files
    file_handler.suffix = "%Y-%m-%d"

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s [%(process)d]: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            file_handler,
        ]
    )

    # # Configure root logger
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format="%(asctime)s [%(levelname)s] %(name)s [%(process)d]: %(message)s",
    #     handlers=[
    #         logging.StreamHandler(sys.stdout),
    #         logging.FileHandler("logs/bot.log", encoding='utf-8'),
    #     ]
    # )

    # Set specific loggers
    logging.getLogger("celery").setLevel(logging.INFO)
    logging.getLogger("celery.worker").setLevel(logging.INFO)
    logging.getLogger("celery.task").setLevel(logging.INFO)

    return logging.getLogger(__name__)