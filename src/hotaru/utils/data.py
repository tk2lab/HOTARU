from collections import namedtuple

import numpy as np


class Data(namedtuple("Data", "imgs mask hz avgx avgt std0 min0 max0 min1 max1")):

    def get_slice(self, start=None, stop=None, step=None):
        s = slice(start, stop, step)
        return self.select(s)

    def select(self, s):
        imgs = self.imgs[s]
        avgx = self.avgx
        avgt = self.avgt[s, np.newaxis, np.newaxis]
        std0 = self.std0
        return (imgs - avgx - avgt) / std0
