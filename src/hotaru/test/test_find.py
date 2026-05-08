from hotaru.data import MovieWithStats
from hotaru.footprint import PeakFinder
from hotaru.footprint import Radius


def test_find():
    data = MovieWithStats.load('sample/imgs.tif', 20.0, batch_size=100)
    radius = Radius(kind='linear', min=2.0, max=8.0, num=10)
    peak_finder = PeakFinder(radius)
    peak_finder.compile()
    peak_finder.fit(data, batch_size=100)
    peakmap = peak_finder.get_peakmap()
    cpeaks, bpeaks = peakmap.reduce(3.0, 7.0, 100, 20)
    print(cpeaks, bpeaks)
