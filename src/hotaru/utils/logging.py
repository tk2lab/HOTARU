import logging

from tqdm import tqdm


class StreamHandler(logging.StreamHandler):

    def emit(self, record):
        try:
            match record.args:
                case ("pbar", "start", desc, total):
                    self.tqdm = tqdm(desc=desc, total=total, file=self.stream)
                case ("pbar", "start", desc, total, postfix):
                    self.tqdm = tqdm(desc=desc, total=total, file=self.stream)
                    self.tqdm.set_postfix_str(postfix)
                case ("pbar", "update", n):
                    self.tqdm.update(n)
                case ("pbar", "update", n, postfix):
                    self.tqdm.set_postfix_str(postfix, refresh=False)
                    self.tqdm.update(n)
                case ("pbar", "close"):
                    self.tqdm.close()
                case _:
                    msg = self.format(record)
                    tqdm.write(msg, end=self.terminator, file=self.stream)
                    self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


class DropPbarFilter(logging.Filter):

    def filter(self, record):
        return (len(record.args) == 0) or (record.args[0] != "pbar")
