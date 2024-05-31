import logging


def set_logger_config():
    logging_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename="all-process.log", format=logging_format, level=logging.INFO)


logger = logging.getLogger(__name__)
set_logger_config()
