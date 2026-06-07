from logging import getLogger
from math import ceil

from keras import ops

from ..ops import gaussian_1d
from ..random import Generator
from ..random import with_rng
from ..typing import Array
from ..typing import Tensor

logger = getLogger(__name__)


@with_rng('num_samples', 'num', 'max_num_spikes')
def sim_spikes(
    num_samples: int,
    num: int,
    max_num_spikes: int,
    intensity_mm: Array | Tensor,
    intensity_mb: Array | Tensor,
    intensity_s: Array | Tensor,
    isi_mm: Array | Tensor,
    isi_ms: Array | Tensor,
    isi_sm: Array | Tensor,
    isi_ss: Array | Tensor,
    random_state: Tensor,
) -> tuple[Tensor, Tensor]:
    rng = Generator(random_state)
    isi = rng.invgauss(
        rng.gamma(isi_mm, isi_ms, shape=num)[:, None],
        rng.gamma(isi_sm, isi_ss, shape=num)[:, None],
        shape=(num, max_num_spikes),
    )
    gs = rng.gamma(
        rng.exponential(intensity_mm, intensity_mb, shape=num)[:, None],
        ops.reshape(intensity_s, (-1, 1)),
        shape=(num, max_num_spikes),
    )

    cell_indices = ops.tile(ops.arange(num, dtype='int32')[:, None], (1, max_num_spikes))
    spike_indices = ops.cumsum(isi, axis=1).astype('int32')
    spikes = ops.zeros((num, num_samples), 'float32').at[cell_indices, spike_indices].add(gs)
    return spikes, rng.state


@with_rng('num_mix')
def sim_dendrites(
    dist_mat: Tensor,
    cell_trs: Tensor,
    kernel: Tensor,
    beta: Tensor,
    num_mix: int,
    random_state: Tensor,
) -> tuple[Tensor, Tensor]:
    rng = Generator(random_state)
    logits = ops.log_softmax(-beta * dist_mat)
    cell_ids = rng.categorical(logits, shape=(num_mix,))
    dend_trs = ops.take_along_axis(cell_trs[None, :, :], cell_ids[:, :, None], axis=1)
    dend_trs = ops.mean(dend_trs, axis=1)[:, :, None]
    dend_trs = ops.conv(dend_trs, kernel[::-1, None, None], 1, 'same', 'channels_last')[:, :, 0]
    return dend_trs, rng.state


def sim_neuropil_traces(
    num: int,
    n_frames: int,
    tau: float,
    rng: Generator,
    hz: float = 1.0,
) -> Tensor:
    tau *= hz
    nd = ceil(3 * tau)
    raw_traces = rng.normal(shape=(num, n_frames + 2 * nd))
    return gaussian_1d(raw_traces, tau, nd=nd)
