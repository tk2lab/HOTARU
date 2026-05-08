from dataclasses import dataclass
from logging import getLogger

from ..saving import PathLike
from .imgs import MovieData
from .stats import Stats
from .stats import StatsCalculator

logger = getLogger(__name__)


@dataclass
class MovieWithStats(MovieData):
    stats: Stats

    @classmethod
    def load(
        cls,
        path: PathLike,
        hz: float,
        cachepath: PathLike | None = None,
        **kwargs,
    ) -> MovieWithStats:
        movie = super().load(path, hz)

        stats = None
        if cachepath is not None:
            try:
                stats = Stats.load(cachepath)
            except Exception:
                stats = None
        if stats is None:
            calculator = StatsCalculator()
            calculator.compile()
            _history = calculator.fit(movie, **kwargs)
            stats = calculator.get_stats()
            if cachepath is not None:
                stats.save(cachepath)

        return MovieWithStats(movie.imgs, movie.mask, movie.hz, stats)
