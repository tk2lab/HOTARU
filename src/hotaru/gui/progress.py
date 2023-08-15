from tqdm import tqdm


class Progress:
    def __init__(self, setter):
        self._setter = setter

    def skip(self):
        pass

    def session(self, name):
        print("session", name)
        self._name = name
        self._setter((self._name, "", "", ""))
        return self

    def set_count(self, total, status=""):
        print("set_count", total, status)
        self._status = status
        self._total = total
        self._n = 0
        self._setter((self._name, str(self._n), str(self._total), str(self._status)))

    def update(self, n, status=""):
        print("update", n, status)
        self._status = status
        self._n += n
        self._setter((self._name, str(self._n), str(self._total), str(self._status)))


class SilentProgress:
    def skip(self):
        pass

    def session(self, name):
        return self

    def set_count(self, total, status=None):
        pass

    def update(self, n, status=None):
        pass


class TQDMProgress:
    def __init__(self):
        self._curr = None

    def skip(self):
        pass

    def session(self, name):
        # print("session", name)
        if self._curr is not None:
            self._curr.close()
        self._name = name
        return self

    def set_count(self, total, status=None):
        # print("set_count", total, status)
        self._curr = tqdm(desc=self._name, total=total)

    def update(self, n, status=None):
        # print("update", n, status)
        if status is not None:
            self._curr.set_postfix_str(status, refresh=False)
        self._curr.update(n)
