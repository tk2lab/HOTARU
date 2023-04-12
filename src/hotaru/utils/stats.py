from collections import namedtuple


class Stats(namedtuple("Stats", "t0 y0 x0 mask avgx avgt std0")):
    def normalize(self, imgs):
        return (self.clip(imgs) - self.avgx - self.avgt[:, None, None]) / self.std0

    def slice(self, start, end, batch):
        return self._replace(avgt=self.avgt[start:end].reshape(batch, -1))

    def clip(self, imgs):
        y0, x0 = self.y0, self.x0
        h, w = self.mask.shape
        return imgs[:, y0 : y0 + h, x0 : x0 + w]
