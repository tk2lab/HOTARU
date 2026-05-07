from dataclasses import dataclass
from logging import getLogger

from ..saving import PathLike
from ..typing import Array
from .data import MovieData
from .stats import StatsCalculator

logger = getLogger(__name__)


@dataclass
class Stats:
    avgx: Array
    avgt: Array
    std0: Array
    min0: Array
    max0: Array
    imin: Array
    imax: Array
    istd: Array
    icor: Array


@dataclass
class MovieWithStats(MovieData):
    stats: Stats

    @classmethod
    def load(cls, path: PathLike, hz: float, **kwargs) -> MovieWithStats:
        movie = super().load(path, hz)
        calculator = StatsCalculator()
        calculator.compile()
        stats = calculator.fit(movie, **kwargs)
        return MovieWithStats(movie.data, movie.mask, movie.hz, Stats(*stats))
