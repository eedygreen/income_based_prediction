
import logging
import os, sys
from datetime import datetime



class ColoredFormatter(logging.Formatter):
    """Custom formater to add colors to log levels"""
    # ANSI color codes
    COLORS = {
        'WARNING': '\033[93m',      # Bold Yellow
        'ERROR': '\033[31m',        # Red
        'CRITICAL': '\033[1;31m',   # Bold Red
        'INFO': '\033[32m',         # Green
        'DEBUG': '\033[36m',        # CYAN
    }

    RESET = '\033[0m'               # Reset to default color

    def __init__(self, fmt=None, datefmt=None, use_colors: bool=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors

    def format(self, record):
        if self.use_colors and record.levelname in self.COLORS:
            original_loglevel = record.levelname
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            result = super().format(record)
            record.levelname = original_loglevel
            return result
        return super().format(record)

class Logger:
    _logger = None
    _instance = None

    # define the singleton pattern
    def __new__(cls):
        if cls._instance is None:
            # New style (Python 3+) supper().__new__(cls)
            cls._instance = super(Logger, cls).__new__(cls)
            cls._setup_logger()
        return cls._instance

    @classmethod
    def _setup_logger(cls):
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # create logger with module name
        cls._logger = logging.getLogger('__name__')
        cls._logger.setLevel(logging.DEBUG)

        # prevent adding handlers multiple times
        if cls._logger.handlers:
            return

        # log format
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        # File format
        log_file = f"logs/app_{datetime.now().strftime('%Y%m%d')}.log"

        # File handler: Add Debug logs to file
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)

        # Console handlers: Stream loglevels except debug logs
        console_handler = logging.StreamHandler(sys.stdout)
        use_colors = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        console_formatter = ColoredFormatter(
            log_format,
            datefmt=date_format,
            use_colors=use_colors
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)

        # add handlers
        cls._logger.addHandler(file_handler)
        cls._logger.addHandler(console_handler)

    def __getattr__(self, name):
        return getattr(self._logger, name)


logger = Logger()
