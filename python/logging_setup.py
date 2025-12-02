import logging
import os
import sys
import time
from datetime import datetime
import traceback

try:
    from colorlog import ColoredFormatter

    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

START_TIME = time.monotonic()


class RustStyleFormatter(logging.Formatter):
    """
    A custom log formatter that mimics the tracing_subscriber format.
    Format: HH:MM:SS +S.msS LEVEL MESSAGE
    """

    level_colors = {
        'DEBUG': '\x1b[34m',  # Blue
        'INFO': '\x1b[32m',  # Green
        'WARNING': '\x1b[33m',  # Yellow
        'ERROR': '\x1b[31m',  # Red
        'CRITICAL': '\x1b[91m',  # Bright Red
        'RESET': '\x1b[0m'
    }

    def __init__(self, tag: str = None, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and COLORLOG_AVAILABLE
        self.tag_str = f"[{tag}] " if tag else ""

    def format(self, record):
        wall_clock_str = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')

        elapsed = time.monotonic() - START_TIME
        secs = int(elapsed)
        millis = int((elapsed - secs) * 1000)
        uptime_str = f"+{secs}.{millis:03d}s"


        level_str = record.levelname

        message_str = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            message_str += "\n" + "".join(traceback.format_exception(*record.exc_info))
        elif record.stack_info:
            message_str += "\n" + record.stack_info

        if self.use_color:
            color = self.level_colors.get(level_str, '')
            reset = self.level_colors['RESET']
            return f"{wall_clock_str} {uptime_str}  {color}{level_str}{reset} {self.tag_str}{message_str}"
        else:
            return f"{wall_clock_str} {uptime_str}  {level_str} {self.tag_str}{message_str}"


def setup_logging(tag: str = None):
    """
    Configures the root logger to behave like the Rust setup.
    """
    # Equivalent to EnvFilter: read level from env var, default to INFO
    log_level_str = os.getenv('PYTHON_LOG_LEVEL', 'INFO').upper()

    log_level = logging.getLevelName(log_level_str)

    is_tty = sys.stdout.isatty()

    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove any existing handlers from the root logger to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = RustStyleFormatter(tag=tag, use_color=is_tty)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
