from dataclasses import dataclass
from logging import getLogger

from .imgs import MovieData
from .stats import Stats
from .stats import StatsCalculator

logger = getLogger(__name__)


@dataclass
class MovieWithStats(MovieData):
    stats: Stats

    def __init__(self, imgs, mask, hz, **kwargs):
        super().__init__(imgs, mask, hz)
        self.stats = StatsCalculator().get_stats(self, **kwargs)
