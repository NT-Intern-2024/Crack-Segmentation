import logging
from .file_utils import check_path_compatibility
from .project import change_to_project_path

def set_logger_config():
    logging_format = "%(asctime)s - %(message)s"
    logging.basicConfig(filename="all-process.log", format=logging_format, level=logging.INFO)

def setup_logger(name: str, log_file: str, level=logging.INFO):
    logger = logging.getLogger(name)

    logger.setLevel(level)

    log_parent_path = "./log"
    check_path_compatibility(log_parent_path)
    
    handler = logging.FileHandler("{}/{}".format(log_parent_path, log_file))
    handler.setLevel(level)

    logging_format = "%(asctime)s | %(levelname)s | [%(filename)s: %(funcName)s()] : %(message)s"
    formatter = logging.Formatter(logging_format)
    handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)

    logger.info(f"------------------------------Setup logger: @{log_file}-------------------------------")
    return logger



# logger = logging.getLogger(__name__)
# set_logger_config()

logger = setup_logger("General", "all-process.log")
logger_classify = setup_logger("Classify", "classify.log")
