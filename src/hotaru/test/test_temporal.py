import numpy as np

from hotaru.config import ComponentProperty
from hotaru.config import TotalProperty
from hotaru.data import MovieWithStats
from hotaru.regularizers import L2Regularizer
from hotaru.regularizers import SparseShapeRegularizer
from hotaru.spatial import FootprintClipper
from hotaru.spatial import PeakFinder
from hotaru.spatial import Radius
from hotaru.temporal import TemporalUpdater
from hotaru.temporal import double_exp_kernel
from hotaru.temporal import exp_kernel


def test_temporal():
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
        cache_path='sample/peak_c.h5',
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

    cell_footprints = footprint_clipper.get_footprints(
        data,
        cell_peaks,
        batch_size=100,
        cache_path='sample/footprint_c.h5',
    )
    back_footprints = footprint_clipper.get_footprints(
        data,
        bg_peaks,
        batch_size=100,
        cache_path='sample/footprint_b.h5',
    )

    cell_props = ComponentProperty(
        np.ones((3, 3), 'float32') / 9,
        SparseShapeRegularizer(nonneg=True),
        lambda _v: 0.0,
        double_exp_kernel(0.08, 0.16, hz=data.hz),
        SparseShapeRegularizer(nonneg=True),
        lambda _a: 0.0,
    )
    back_props = ComponentProperty(
        np.ones((3, 3), 'float32') / 9,
        L2Regularizer(nonneg=False),
        lambda v: 10.0 * np.sum(np.square(v)),
        exp_kernel(1.0, hz=data.hz),
        L2Regularizer(nonneg=True),
        lambda a: 10.0 * np.sum(np.square(a)),
    )
    props = TotalProperty(
        [cell_props, back_props],
        L2Regularizer(10.0),
        L2Regularizer(10.0),
    )
    updater = TemporalUpdater(props)
    updater.prepare(
        data,
        cell_footprints,
        back_footprints,
        batch_size=100,
        names=['c', 'b'],
        cache_path='sample/scorr.h5',
        force=True,
    )
    updater.compile(learning_rate=1e-6, nesterov=30.0)
    updater.update(
        steps_per_epoch=100,
        epochs=100,
        early_stopping={'min_delta': 1e-6, 'patience': 3},
        names=['c', 'b'],
        cache_path='sample/temporal.h5',
        force=True,
    )
