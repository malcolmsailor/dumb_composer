import logging
import os


def configure_logging(log_file_path, log_level, append_to_log):
    if log_file_path is None:
        return
    if not append_to_log:
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
    loglevel = getattr(logging, log_level.upper())
    logging.basicConfig(filename=log_file_path, level=loglevel)
