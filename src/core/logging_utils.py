"""Project-wide logging helpers.

Uses stdout-only, concise format, level from LOG_LEVEL env (default INFO), and
standardizes uvicorn access logs to the same format.
"""

import logging
import os
from logging.config import dictConfig


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging() -> None:
    """Configure root logging once.

    - Level: LOG_LEVEL env (DEBUG/INFO/WARNING/ERROR/CRITICAL), default INFO.
    - Output: stdout only (Docker/Streamlit-friendly).
    - Format: time, level, logger name, message.
    """

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": LOG_FORMAT,
                    "datefmt": DATE_FORMAT,
                }
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                }
            },
            "root": {"level": level, "handlers": ["stdout"]},
            # Align common libraries to root level unless overridden elsewhere
            "loggers": {
                "uvicorn": {"level": level, "handlers": ["stdout"], "propagate": False},
                "uvicorn.error": {"level": level, "handlers": ["stdout"], "propagate": False},
                "uvicorn.access": {"level": level, "handlers": ["stdout"], "propagate": False},
                "fastapi": {"level": level, "handlers": ["stdout"], "propagate": False},
                # reduce noisy client libs by default
                "httpx": {"level": logging.WARNING, "handlers": ["stdout"], "propagate": False},
                "openai": {"level": logging.INFO, "handlers": ["stdout"], "propagate": False},
                "watchdog": {"level": logging.WARNING, "handlers": ["stdout"], "propagate": False},
            },
        }
    )
