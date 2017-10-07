import logging

__version__ = '0.2.2'

class NullHandler(logging.Handler):
    #For python versions <= 2.6; same as `logging.NullHandler` in 2.7.
    def emit(self, record):
        pass

logger = logging.getLogger('tmlib')
if len(logger.handlers) == 0:	# To ensure reload() doesn't add another one
    logger.addHandler(NullHandler())

__all__ = ['lda', 'datasets', 'preprocessing', 'visualization']
