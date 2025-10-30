import logging
import sys


def configure_logging(level: str = "INFO") -> None:
	root = logging.getLogger()
	root.setLevel(level.upper())

	if not root.handlers:
		stream_handler = logging.StreamHandler(stream=sys.stdout)
		formatter = logging.Formatter(
			fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
			datefmt="%Y-%m-%dT%H:%M:%S%z",
		)
		stream_handler.setFormatter(formatter)
		root.addHandler(stream_handler)


