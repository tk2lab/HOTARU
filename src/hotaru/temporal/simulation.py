from collections.abc import Sequence
from functools import partial
from functools import wraps
from math import ceil

from keras import ops
from keras.backend import backend
from tqdm import trange

from ..ops import gaussian_1d
from ..random import Generator
from ..typing import Tensor
from .dynamics import Kernel
from .trace import Traces


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
    upsample_factor: int,
    kernel: Kernel,
    isi_mm: Sequence[float],
    isi_ms: Sequence[float],
    isi_sm: Sequence[float],
    isi_ss: Sequence[float],
    intensity_s: Sequence[float],
    rng: Generator,
) -> Traces:
    hz_high = upsample_factor * kernel.hz
    num_samples_high = upsample_factor * num_samples + kernel.pad_size(upsample_factor)
    max_num_spikes = ceil(2 * num_samples)

    num = len(isi_mm)
    spikes = ops.zeros((num, num_samples_high), 'float32')
    for i in trange(num, ncols=150, desc='make traces'):
        spki = sim_spike_train(
            hz_high,
            num_samples_high,
            max_num_spikes,
            rng.invgauss(isi_mm[i], isi_ms[i]),
            rng.invgauss(isi_sm[i], isi_ss[i]),
            ops.convert_to_tensor(intensity_s[i]),
            rng=rng,
        )
        spikes = spikes.at[i].add(spki)

    obs = kernel(spikes, upsample_factor=upsample_factor)[:, ::upsample_factor]
    return Traces(obs)


@with_rng('hz', 'num_samples', 'max_num')
def sim_spike_train(
    hz: float,
    num_samples: int,
    max_num: int,
    isi_m: Tensor,
    isi_s: Tensor,
    intensity_s: Tensor,
    random_state: Tensor,
) -> tuple[Tensor, Tensor]:
    rng = Generator(random_state)
    isi = rng.invgamma(isi_m, isi_s, shape=(max_num,))
    gs = rng.invgauss(1, intensity_s, shape=(max_num,))
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
    dend_trs = ops.sum(dend_trs, axis=1)[:, :, None]
    dend_trs = ops.conv(dend_trs, kernel[::-1, None, None], 1, 'same', 'channels_last')[:, :, 0]
    return dend_trs, rng.state


def sim_neuropil_traces(
    num: int,
    n_frames: int,
    tau: float,
    rng: Generator,
    hz: float = 1.0,
) -> Traces:
    tau *= hz
    nd = ceil(3 * tau)
    raw_traces = rng.normal(shape=(num, n_frames + 2 * nd))
    traces = gaussian_1d(raw_traces, tau, nd=nd)
    return Traces(traces)
