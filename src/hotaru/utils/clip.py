from collections import namedtuple

import numpy as np


def get_clip(shape, clip):
    h, w = shape
    match clip:
        case [Clip(), *_]:
            return clip
        case None:
            return [Clip(0, h, 0, w, 0)]
        case {"type": "Div", "ynum": ynum, "xnum": xnum, "margin": margin}:
            dy = (h - 2 * margin + ynum - 1) // ynum
            dx = (w - 2 * margin + xnum - 1) // xnum
            clips = []
            for py in range(ynum):
                if py == 0:
                    y0 = 0
                else:
                    y0 = margin + dy * py
                if py == ynum - 1:
                    y1 = h
                else:
                    y1 = margin + dy * (py + 1)
                for px in range(xnum):
                    if px == 0:
                        x0 = 0
                    else:
                        x0 = margin + dx * px
                    if px == xnum - 1:
                        x1 = w
                    else:
                        x1 = margin + dx * (px + 1)
                    clips.append(Clip(y0, y1, x0, x1, margin))
            return clips
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

    def in_clipping_area(self, y, x):
        return (self.y0 <= y) & (y < self.y1) & (self.x0 <= x) & (x < self.x1)
