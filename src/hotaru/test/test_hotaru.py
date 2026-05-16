import numpy as np
import pytest

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


def test_data():
    MovieWithStats.load(
        'sample/imgs.tif',
        hz=20.0,
        batch_size=100,
        cache_path='sample/stats.h5',
        force=True,
    )


@pytest.fixture
def data():
    return MovieWithStats.load(
        'sample/imgs.tif',
        hz=20.0,
        batch_size=100,
        cache_path='sample/stats.h5',
    )


@pytest.fixture
def peak_finder():
    return PeakFinder()


radius = {
    'c': Radius(kind='logscale', min=2.0, max=8.0, num=10),
    'b': Radius(kind='logscale', min=7.0, max=20.0, num=10),
}


@pytest.mark.parametrize('radius,label', [(radius, label) for label, radius in radius.items()])
def test_peakmap(peak_finder, data, radius, label):
    peak_finder.get_peakmap(
        data,
        radius,
        batch_size=100,
        cache_path=f'sample/peakmap_{label}.h5',
        force=True,
    )


@pytest.fixture
def peakmap_factory(peak_finder, data):
    def _create(label):
        return peak_finder.get_peakmap(
            data,
            radius[label],
            batch_size=100,
            cache_path=f'sample/peakmap_{label}.h5',
        )
    return _create


@pytest.mark.parametrize('label', list(radius.keys()))
def test_peaks(peakmap_factory, label):
    peakmap = peakmap_factory(label)
    peakmap.reduce(
        min_distance_ratio=2.0,
        block_size=100,
        cache_path=f'sample/peak_{label}.h5',
        force=True,
    )


@pytest.fixture
def peaks_factory(peakmap_factory):
    def _create(label):
        peakmap = peakmap_factory(label)
        return peakmap.reduce(
            min_distance_ratio=2.0,
            block_size=100,
            cache_path=f'sample/peak_{label}.h5',
        )
    return _create


@pytest.fixture
def footprint_clipper():
    return FootprintClipper()


@pytest.mark.parametrize('label', list(radius.keys()))
def test_clip(footprint_clipper, data, peaks_factory, label):
    peaks = peaks_factory(label)
    footprint_clipper.get_footprints(
        data,
        peaks,
        batch_size=100,
        cache_path=f'sample/footprint_{label}.h5',
        force=True,
    )


@pytest.fixture
def footprints_factory(footprint_clipper, data, peaks_factory):
    def _create(label):
        peaks = peaks_factory(label)
        return footprint_clipper.get_footprints(
            data,
            peaks,
            batch_size=100,
            cache_path=f'sample/footprint_{label}.h5',
        )
    return _create


@pytest.fixture
def cell_props(data):
    return ComponentProperty(
        np.ones((3, 3), 'float32') / 9,
        SparseShapeRegularizer(nonneg=True),
        lambda _v: 0.0,
        double_exp_kernel(0.08, 0.16, hz=data.hz),
        SparseShapeRegularizer(nonneg=True),
        lambda _a: 0.0,
    )


@pytest.fixture
def back_props(data):
    return ComponentProperty(
        np.ones((3, 3), 'float32') / 9,
        L2Regularizer(nonneg=False),
        lambda v: 10.0 * np.sum(np.square(v)),
        exp_kernel(1.0, hz=data.hz),
        L2Regularizer(nonneg=True),
        lambda a: 10.0 * np.sum(np.square(a)),
    )


@pytest.fixture
def props(cell_props, back_props):
    return TotalProperty(
        [cell_props, back_props],
        L2Regularizer(10.0),
        L2Regularizer(10.0),
    )


@pytest.fixture
def temporal_updater(props):
    return TemporalUpdater(props)


def test_temporal(temporal_updater, data, footprints_factory):
    labels = 'c', 'b'
    temporal_updater.prepare(
        data,
        *(footprints_factory(label) for label in labels),
        batch_size=100,
        names=labels,
        cache_path='sample/scorr.h5',
        force=True,
    )
    temporal_updater.compile(learning_rate=1e-6, nesterov=30.0)
    temporal_updater.update(
        steps_per_epoch=100,
        epochs=100,
        early_stopping={'min_delta': 1e-6, 'patience': 3},
        names=labels,
        cache_path='sample/temporal.h5',
        force=True,
    )
