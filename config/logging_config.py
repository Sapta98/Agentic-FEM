import logging
import os


def configure_logging(level: str | None = None) -> None:
	"""Configure root logging with a consistent, concise format.

	Level precedence: function arg > ENV LOG_LEVEL > INFO.
	"""
	level_name = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
	try:
		log_level = getattr(logging, level_name, logging.INFO)
	except Exception:
		log_level = logging.INFO

	logging.basicConfig(
		level=log_level,
		format="%(asctime)s %(levelname)s %(name)s: %(message)s",
	)


def get_logger(name: str) -> logging.Logger:
	"""Get a module logger."""
	return logging.getLogger(name)


