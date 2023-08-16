from collections import namedtuple

import numpy as np


def get_clip(shape, *clip):
    h, w = shape
    match clip:
        case (None,):
            return [Clip(0, h, 0, w, 0)]
        case ("Div", numy, numx, margin):
            dy = (h + numy - 1) // numy
            dx = (w + numx - 1) // numx
            return [
                Clip(py * dy, (py + 1) * dy, px * dx, (px + 1) * dx, margin)
                for py in range(numy)
                for px in range(numx)
            ]
        case _:
            raise ValueError()


class Clip(namedtuple("Clip", "y0 y1 x0 x1 margin")):
    def range(self, h, w):
        y0 = max(self.y0 - self.margin, 0)
        y1 = min(self.y1 + self.margin, h)
        x0 = max(self.x0 - self.margin, 0)
        x1 = min(self.x1 + self.margin, w)
        return y0, y1, x0, x1

    def clip(self, img):
        if img is None:
            return img
        else:
            h, w = img.shape[-2:]
            y0, y1, x0, x1 = self.range(h, w)
            return img[..., y0:y1, x0:x1]

    def unclip(self, val, mask, shape):
        nk = val.shape[0]
        h, w = shape
        y0, y1, x0, x1 = self.range(h, w)
        if mask is None:
            tmp = val.reshape(nk, y1 - y0, x1 - x0)
        else:
            tmp = np.zeros((nk, y1 - y0, x1 - x0), val.dtype)
            tmp[:, mask] = val
        out = np.zeros((nk, h, w), val.dtype)
        out[:, y0:y1, x0:x1] = tmp
        return out

    def is_active(self, y, x):
        return (self.y0 <= y) & (y < self.y1) & (self.x0 <= x) & (x < self.x1)
