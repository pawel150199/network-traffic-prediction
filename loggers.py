from typing import Any
import colorlog
import argparse
import logging

def configureLogger() -> Any:
    """
    Function configure logger

    Returns:
        Any: Logger from colorlog class
    
    Usage::

        from loggers import configureLogger

        logger = configureLogger()
        logger.info("Logger is working fine!")

    """  

    logger = colorlog.getLogger()
    logger.setLevel(logging.INFO)

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s - %(message)s",
            log_colors={
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
    )
    logger.addHandler(handler)
    return  logger