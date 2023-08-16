from collections import namedtuple
from logging import getLogger

import numpy as np

logger = getLogger(__name__)


class Data(namedtuple("Data", "imgs mask hz avgx avgt std0 min0 max0 min1 max1")):

    @property
    def nt(self):
        return self.imgs.shape[0]

    @property
    def ns(self):
        match self.mask:
            case None:
                return self.shape[0] * self.shape[1]
            case _:
                return np.count_nonzero(self.mask)

    @property
    def shape(self):
        return self.imgs.shape[1:]

    def clip(self, clip):
        return self._replace(
            imgs=clip(self.imgs),
            mask=clip(self.mask),
            avgx=clip(self.avgx),
        )

    def apply_mask(self, x, mask_type=np.nan):
        match (mask_type, self.mask):
            case (False, _):
                pass
            case (True, None):
                *shape, h, w = x.shape
                x = x.reshape(shape + [h * w])
            case (True, mask):
                x = x[..., mask]
            case (val, None):
                pass
            case (val, mask):
                x[..., mask] = val
            case _:
                raise ValueError()
        return x

    def select(self, s, mask_type=np.nan):
        imgs = self.imgs[s]
        avgx = self.avgx
        avgt = self.avgt[s, np.newaxis, np.newaxis]
        std0 = self.std0
        imgs = (imgs - avgx - avgt) / std0
        return self.apply_mask(imgs, mask_type)

    def data(self, idx=None, mask_type=np.nan):
        if idx is None:
            idx = range(self.nt)
        for i in idx:
            yield self.select(i, mask_type)

    def datax(self):
        data = self.imgs.reshape(self.nt, -1)
        avgx = self.avgx.ravel()
        avgt = self.avgt
        std0 = self.std0
        if self.mask is None:
            for d, a in zip(data.T, avgx):
                yield (d - a - avgt) / std0
        else:
            mask = self.mask.ravel()
            for m, d, a in zip(mask, data.T, avgx):
                if m:
                    yield (d - a - avgt) / std0
