from collections.abc import Sequence
from functools import partial
from functools import wraps
from logging import getLogger
from math import ceil

from keras import ops
from keras.backend import backend
from tqdm import trange

from ..ops import gaussian_1d
from ..random import Generator
from ..typing import Tensor
from .dynamics import Kernel

logger = getLogger(__name__)


def with_rng(*static_argnames):
    def dec(fn):
        @wraps(fn)
        def wrap_func(*args, rng: Generator, **kwargs) -> Tensor:
            func = fn
            if backend() == 'jax':
                import jax

                func = partial(jax.jit, static_argnames=static_argnames)(fn)
            val, state = func(*args, **kwargs, random_state=rng.state)
            rng.state = state
            return val

        return wrap_func

    return dec


def sim_traces(
    num_samples: int,
    kernel: Kernel,
    upsample_factor: int,
    intensity_m: Sequence[float],
    intensity_b: Sequence[float],
    intensity_s: Sequence[float],
    isi_mm: Sequence[float],
    isi_ms: Sequence[float],
    isi_sm: Sequence[float],
    isi_ss: Sequence[float],
    rng: Generator,
) -> Tensor:
    max_num_spikes = ceil(2 * num_samples)

    num = len(isi_mm)
    num_samples_with_pad = num_samples + kernel.pad_size(upsample_factor)
    spikes = ops.zeros((num, num_samples_with_pad), 'float32')
    intensity_m = ops.convert_to_tensor(intensity_m)
    intensity_b = ops.convert_to_tensor(intensity_b)
    u = ops.exp(-intensity_b / intensity_m) * rng.uniform(shape=num)
    intensity_m = -intensity_m * ops.log(u)
    intensity_s = ops.convert_to_tensor(intensity_s)
    logger.info('intensity: %s %s %s', intensity_m.min(), intensity_m.max(), intensity_m.mean())
    for i in trange(num, ncols=150, desc='make traces'):
        spki = sim_spike_train(
            kernel.hz,
            num_samples_with_pad,
            max_num_spikes,
            intensity_m[i],
            intensity_s[i],
            rng.invgauss(isi_mm[i], isi_ms[i]),
            rng.invgauss(isi_sm[i], isi_ss[i]),
            rng=rng,
        )
        spikes = spikes.at[i].set(spki)
    logger.info('spikes: %s %s %s', spikes.min(), spikes.max(), ops.count_nonzero(spikes))

    return kernel(spikes, upsample_factor=upsample_factor)[:, ::upsample_factor]


@with_rng('hz', 'num_samples', 'max_num')
def sim_spike_train(
    hz: float,
    num_samples: int,
    max_num: int,
    intensity_m: Tensor,
    intensity_s: Tensor,
    isi_m: Tensor,
    isi_s: Tensor,
    random_state: Tensor,
) -> tuple[Tensor, Tensor]:
    rng = Generator(random_state)
    isi = rng.invgamma(isi_m, isi_s, shape=(max_num,))
    gs = rng.invgauss(intensity_m, intensity_s, shape=(max_num,))
    spike_indices = (ops.cumsum(isi) * hz).astype('int32')
    spike_train = ops.zeros(num_samples, 'float32').at[spike_indices].set(gs)
    return spike_train, rng.state


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
