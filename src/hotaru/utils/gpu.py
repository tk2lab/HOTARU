import os
import math
import subprocess
import psutil
from logging import getLogger

import tensorflow as tf
import jax
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import PositionalSharding
from jax.dlpack import from_dlpack

logger = getLogger(__name__)


def from_tf(x_tf):
    x_dl = tf.experimental.dlpack.to_dlpack(x_tf)
    return from_dlpack(x_dl)


def get_gpu_env(env):
    match env:
        case GpuEnv():
            return env
        case None:
            return GpuEnv()
        case _:
            return GpuEnv(**env)


class GpuEnv:
    def __init__(self, num_devices=-1, memsize=-1, label="gpu"):
        if not (num_devices > 0 and memsize > 0):
            gpu_info = get_gpu_info()
            if not (num_devices > 0):
                num_devices = len(gpu_info)
            if not (memsize > 0):
                memsize = int(min(gpu_info[: num_devices]))
        self.num_devices = num_devices
        self.memsize = memsize * 1e6
        self.label = label

    def batch(self, factor, size):
        n = self.num_devices
        batch_simple = min(size, max(n, int(n * self.memsize / factor)))
        batch_padded = n * ((batch_simple + n - 1) // n)
        return batch_padded

    def batch_sqrt(self, factor, size):
        n = self.num_devices
        batch_size = min(size, max(n, int(n * math.sqrt(self.memsize / factor))))
        batch = n, (batch_size + n - 1) // n
        logger.debug("batch: %s %s", (n, self.memsize, factor, size), batch)
        return batch

    def sharding(self, shape):
        if shape is None:
            shape = (self.num_devices,)
        return PositionalSharding(create_device_mesh(shape))


def get_gpu_info():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        encoding="utf-8",
        check=True,
    )
    return [int(x) for x in result.stdout.strip().split(os.linesep)]


def get_gpu_used():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        encoding="utf-8",
        check=True,
    )
    return [int(x) for x in result.stdout.strip().split(os.linesep)]


def get_xla_stats():
    backend = jax.lib.xla_bridge.get_backend()
    lbs = backend.live_buffers()
    les = backend.live_executables()
    mem = psutil.Process().memory_info().rss
    return dict(mem=mem * 1e-6, executable=len(les), buffer=[lb.shape for lb in lbs])


def delete_xla_buffers():
    backend = jax.lib.xla_bridge.get_backend()
    for buf in backend.live_buffers():
        buf.delete()
    #for exe in backend.live_executables():
    #    exe.delete()
