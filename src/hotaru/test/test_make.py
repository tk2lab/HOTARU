from hotaru.data import MovieWithStats
from hotaru.footprint import FootprintClipper
from hotaru.footprint import PeakFinder
from hotaru.footprint import Radius


def test_make():
    data = MovieWithStats.load(
        'sample/imgs.tif',
        hz=20.0,
        batch_size=100,
        cache_path='sample/stats.h5',
    )

    peak_finder = PeakFinder()
    footprint_clipper = FootprintClipper()

    cell_peaks = peak_finder.get_peakmap(
        data,
        Radius(kind='logscale', min=2.0, max=8.0, num=10),
        batch_size=100,
        cache_path='sample/peakmap_c.h5',
    ).reduce(
        min_distance_ratio=2.0,
        block_size=100,
        cache_path='sample/paak_c.h5',
    )
    bg_peaks = peak_finder.get_peakmap(
        data,
        Radius(kind='logscale', min=7.0, max=20.0, num=10),
        batch_size=100,
        cache_path='sample/peakmap_b.h5',
    ).reduce(
        min_distance_ratio=2.0,
        block_size=100,
        cache_path='sample/peak_b.h5',
    )

    cell_footprints = footprint_clipper.get_footprints(data, cell_peaks, batch_size=100)
    print(cell_footprints)
    bg_footprints = footprint_clipper.get_footprints(data, bg_peaks, batch_size=100)
    print(bg_footprints)
