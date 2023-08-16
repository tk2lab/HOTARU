from collections import namedtuple


def get_range(w, num, pos, margin):
    size = (w + (num - 1) * margin) // num
    stride = size - margin
    x0 = w - size if pos == num - 1 else pos * stride
    x1 = None if  pos == num - 1 else x0 + size
    return x0, x1


def get_clip(shape, *clip):
    print(clip)
    match clip:
        case (None,):
            return [Clip(None, None, None, None)]
        case ("Div", numy, numx, margin):
            h, w = shape
            out = []
            for posy in range(numy):
                for posx in range(numx):
                    y0, y1 = get_range(h, numy, posy, margin)
                    x0, x1 = get_range(w, numx, posx, margin)
                    out.append(Clip(y0, y1, x0, x1))
            return out
        case _:
            return [Clip(*clip)]


class Clip(namedtuple("Clip", "y0 y1 x0 x1")):
    def __call__(self, img):
        if img is None:
            return img
        else:
            slicey = slice(self.y0, self.y1)
            slicex = slice(self.x0, self.x1)
            return img[..., slicey, slicex]
