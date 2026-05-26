import numpy as np
import scipy.stats as st

from .dynamics import Kernel
from .trace import Traces


def sim_single_trace(
    kernel: Kernel,
    hz: float,
    length: float,
    isi_mean: float,
    isi_shape: float,
    upsample_factor: int,
    rng: np.random.Generator,
):
    hz_high = hz * upsample_factor
    num_samples_high = int(np.ceil(length * hz_high))

    isi_dist = st.lognorm(isi_shape, scale=isi_mean)
    isi = isi_dist.rvs(int(2 * length / isi_mean), random_state=rng)
    ts = np.cumsum(isi)
    ts = ts[ts < length]

    spike_train_high = np.zeros(num_samples_high, dtype='float32')
    bins_high = (ts * hz_high).astype(np.int32)
    bins_high = bins_high[bins_high < num_samples_high]
    np.add.at(spike_train_high, bins_high, 1.0)

    obs_high = kernel.apply(spike_train_high, hz_high)
    obs = obs_high[::upsample_factor].astype('float32')
    return obs


def sim_traces(
    kernel: Kernel,
    hz: float,
    length: float,
    isi_mean_mean: float,
    isi_mean_shape: float,
    isi_shape_mean: float,
    isi_shape_shape: float,
    num: int,
    upsample_factor: int,
    alpha: float = 1e-2,
    rng_or_seed: np.random.Generator | int | None = None,
) -> Traces:
    match rng_or_seed:
        case np.random.Generator() as rng:
            pass
        case seed:
            rng = np.random.default_rng(seed)

    num_samples_low = int(length * hz)
    obs = np.empty((num, num_samples_low), 'float32')

    isi_mean = st.invgamma(
        isi_mean_shape,
        scale=isi_mean_mean / isi_mean_shape,
    ).rvs(num, random_state=rng)
    isi_shape = st.invgamma(
        isi_shape_shape,
        scale=isi_shape_mean / isi_shape_shape,
    ).rvs(num, random_state=rng)
    for i in range(num):
        obs[i] = sim_single_trace(kernel, hz, length, isi_mean[i], isi_shape, upsample_factor, rng)

    return Traces(obs)
