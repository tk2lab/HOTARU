class SimpleProgress:
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
