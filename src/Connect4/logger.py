import utils
import logging

from config import Config

config = Config()

def setup_logger(name):
    logging.basicConfig(format=config.Logger_Format)
    logger = logging.getLogger(name)
    logger.setLevel(config.Logger_Level)
    return logger


logger = setup_logger(name='Connect4')


def info(*msg):
    logger.info(*msg)


def debug(*msg):
    logger.debug(*msg)