import os
import logging
import datetime


def init_logger(LOG_PATH="./log"):
    os.makedirs(LOG_PATH, exist_ok=True)
    today = datetime.date.today().strftime('%m%d')
    log_filename = LOG_PATH + f'/log_{today}.log'
    formatter = logging.Formatter('[%(levelname)s] [%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger("log")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return
