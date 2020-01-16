import logging
import os
import sys
from enum import IntEnum
import tensorflow as tf

from network_flow_generator import __name__ as program_name

try:
    import colorlog
    HAVE_COLORLOG = True
except ImportError:
    HAVE_COLORLOG = False


class LogLevel(IntEnum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class Logger:

    _root_logger = logging.getLogger(program_name)

    @classmethod
    def configure(cls, loglevel=LogLevel.INFO):
        """Configures the logger.

        Args:
            loglevel (int): The logging level.
        """
        cls._root_logger.setLevel(loglevel)

        if loglevel == LogLevel.DEBUG:
            format_str = "%(asctime)s %(levelname)-8s [%(pathname)s:%(lineno)d]: %(message)s"
        else:
            format_str = "%(asctime)s %(levelname)-8s: %(message)s"
            # set logging for tensorflow
            tf.get_logger().setLevel(logging.ERROR)

        if HAVE_COLORLOG and os.isatty(1):
            format_str = "%(log_color)s" + format_str
            formatter = colorlog.ColoredFormatter(
                format_str,
                datefmt=None,
                reset=True,
                log_colors={
                    "DEBUG": "reset",
                    "INFO": "reset",
                    "WARNING": "bold_yellow",
                    "ERROR": "bold_red",
                    "CRITICAL": "bold_red"
                },
                secondary_log_colors={},
                style="%")

        else:
            formatter = logging.Formatter(format_str)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        cls._root_logger.addHandler(stream_handler)

    @classmethod
    def get(cls) -> logging.Logger:
        return cls._root_logger
