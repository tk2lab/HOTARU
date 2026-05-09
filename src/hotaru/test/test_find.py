from hotaru.data import MovieWithStats
from hotaru.footprint import PeakFinder
from hotaru.footprint import Radius


def test_find():
    data = MovieWithStats.load(
        'sample/imgs.tif',
        hz=20.0,
        batch_size=100,
        cache_path='sample/stats.h5',
    )

    peak_finder = PeakFinder()

    cell_radius = Radius(kind='logscale', min=2.0, max=8.0, num=10)
    cell_peakmap = peak_finder.get_peakmap(data, cell_radius, batch_size=100)
    cell_peaks = cell_peakmap.reduce(min_distance_ratio=2.0, block_size=100)
    print(cell_peaks)

    bg_radius = Radius(kind='logscale', min=7.0, max=20.0, num=10)
    bg_peakmap = peak_finder.get_peakmap(data, bg_radius, batch_size=100)
    bg_peaks = bg_peakmap.reduce(min_distance_ratio=2.0, block_size=100)
    print(bg_peaks)
