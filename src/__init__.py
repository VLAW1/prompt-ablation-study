"""
Top level __init__ for the project.
"""

__version__ = '0.1.0'

__all__ = ['__version__']


import importlib


def _initialize():
    # Lazy initialization of logging configuration
    log_module = importlib.import_module('src.logging')
    log_module.log.info('Initializing...')


_initialize()
