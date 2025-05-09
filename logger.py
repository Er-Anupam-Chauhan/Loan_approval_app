# logger.py
import logging

def get_logger(log_file='logs/app.log', level=logging.INFO):

    """
    Creates and returns a logger instance with the specified log file and level.

    Parameters:
    log_file (str): Path to the log file. Defaults to 'logs/app.log'.
    level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
    logging.Logger: Configured logger instance.
    """
        
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Avoid duplicate log entries
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter( '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
