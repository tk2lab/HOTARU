from tqdm import tqdm


def get_progress(pbar):
    match pbar:
        case Progress():
            return pbar
        case "silent" | False | None:
            return Progress()
        case "simple":
            return ConsoleProgress()
        case "tqdm":
            return TQDMProgress()


class Progress:
    def skip(self):
        pass

    def session(self, name, total, status=None):
        pass

    def update(self, n, status=None):
        pass

    def close(self):
        pass


class ConsoleProgress(Progress):
    def __init__(self, setter):
        self._setter = setter

    def skip(self):
        pass

    def session(self, name, total, status=None):
        print("session", name)
        self._name = name
        self._setter((self._name, "", "", ""))
        print("set_count", total, status)
        self._status = status
        self._total = total if total > 0 else None
        self._n = 0
        self._setter((self._name, str(self._n), str(self._total), str(self._status)))

    def update(self, n, status=""):
        print("update", n, status)
        self._status = status
        self._n += n
        self._setter((self._name, str(self._n), str(self._total), str(self._status)))


class TQDMProgress(Progress):
    def __init__(self):
        self._curr = None

    def skip(self):
        pass

    def session(self, name, total, status=None):
        self.close()
        self._name = name
        self._curr = tqdm(desc=self._name, total=total)
        if status is not None:
            self._curr.set_postfix_str(status, refresh=False)

    def update(self, n, status=None):
        if status is not None:
            self._curr.set_postfix_str(status, refresh=False)
        self._curr.update(n)

    def close(self):
        if self._curr is not None:
            self._curr.close()
        self._curr = None


"""
class SimpleProgress(Progress):
    def __init__(self, total=None):
        self.reset(total)

    def reset(self, total):
        self.total = total
        self.n = 0

    def update(self, n):
        self.n += n

    @property
    def value(self):
        return 100 * self.n / self.total

    @property
    def label(self):
        return f"{self.n} / {self.total}"
"""
