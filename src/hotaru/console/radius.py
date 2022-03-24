from cleo import option
import numpy as np


class RadiusMixin:

    _options = [
        option('radius-kind', 'k', '{linear,log,manual}', False, False, False, 'log'),
        option('radius-min', 'i', 'radius of cell (px)', False, False, False, 2.0),
        option('radius-max', 'a', 'radius of cell (px)', False, False, False, 24.0),
        option('radius-num', 'u', '', False, False, False, 13),
        option('radius', 'r', '', False, True, True),
    ]

    def radius(self):
        kind = self.option('radius-kind')
        if kind == 'manual':
            return np.array(
                [float(v) for v in self.option('radius')], dtype=np.float32,
            )
        else:
            rmin = float(self.option('radius-min'))
            rmax = float(self.option('radius-max'))
            rnum = int(self.option('radius-num'))
            if kind == 'linear':
                return np.linspace(rmin, rmax, rmin, dtype=np.float32)
            elif kind == 'log':
                return np.logspace(
                    np.log10(rmin), np.log10(rmax), rnum, dtype=np.float32,
                )
        self.line(f'<error>bad radius kind</error>: {kind}')
