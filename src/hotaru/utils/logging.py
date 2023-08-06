import logging

from tqdm import tqdm


logger = logging.getLogger("hotaru")


class LoggingHandler(logging.StreamHandler):

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator, file=self.stream)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)
