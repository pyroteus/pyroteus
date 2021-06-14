"""
Loggers for Pyroteus.

Code mostly copied from [the Thetis project](https://thetisproject.org).
"""
import firedrake
import logging


__all__ = ['logger', 'output_logger', 'debug', 'info', 'warning', 'error', 'critical', 'pyrint', 'set_log_level']


def get_new_logger(name, fmt='%(levelname)s %(message)s'):
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    if firedrake.COMM_WORLD.rank != 0:
        handler = logging.NullHandler()
    logger.addHandler(handler)
    return logger


logger = get_new_logger('pyroteus')
logger.setLevel(logging.WARNING)
log = logger.log
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

output_logger = get_new_logger('pyroteus_output', fmt='%(message)s')
output_logger.setLevel(logging.INFO)
pyrint = output_logger.info


def set_log_level(level):
    firedrake.set_log_level(level)
    logger.setLevel(level)
