from collections import namedtuple


def get_clip(clip, shape=None):
    match clip:
        case Clip():
            return clip
        case None:
            return Clip(None, None, None, None)
        case ["HDiv", num, pos, margin]:
            h, w = shape
            size = (w + (num - 1) * margin) // num
            stride = size - margin
            x0 = pos * stride
            x1 = x0 + size
            return Clip(None, None, x0, x1)
        case _:
            return Clip(*clip)


class Clip(namedtuple("Clip", "y0 y1 x0 x1")):
    def __call__(self, img):
        if img is None:
            return img
        else:
            slicey = slice(self.y0, self.y1)
            slicex = slice(self.x0, self.x1)
            return img[..., slicey, slicex]
