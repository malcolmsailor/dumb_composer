import logging
import pdb
import sys
import traceback


def custom_excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
    pdb.post_mortem(exc_traceback)


def setup_logging(debug_level):
    # Map debug level string to logging level
    debug_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    # Set up the logging system
    logging.basicConfig(level=debug_levels.get(debug_level, logging.WARNING))
