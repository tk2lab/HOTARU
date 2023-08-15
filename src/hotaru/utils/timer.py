import time

from logging import getLogger

default_logger = getLogger(__name__)


class Timer:
    def __init__(self, name=None, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.time = time.time()
        self.ptime = time.process_time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.time = time.time() - self.time
        self.ptime = time.process_time() - self.ptime
        if self.logger:
            logger = self.logger
        else:
            logger = default_logger
        logger.debug("%s: %f %f", self.name, self.time, self.ptime)

    def get(self):
        return self.time, self.ptime
