import os
import subprocess
from logging import getLogger

logger = getLogger(__name__)


class GpuEnv:
    def __init__(self, num_devices=None, memsize=None, label=None):
        if not (num_devices > 0 and memsize > 0):
            gpu_info = get_gpu_info()
            logger.info("gpu %s", gpu_info)
            if not (num_devices > 0):
                num_devices = len(gpu_info)
            if not (memsize > 0):
                memsize = int(min(gpu_info[: num_devices]))
        self.num_devices = num_devices
        self.memsize = memsize * 1e6
        self.label = label

    def batch(self, factor, size):
        n = self.num_devices
        batch_size = min(size, max(n, int(n * self.memsize / factor)))
        batch = n, (batch_size + n - 1) // n
        logger.info("batch: %s %s", (n, self.memsize, factor, size), batch)
        return batch


def get_gpu_env(env):
    if isinstance(env, GpuEnv):
        return env
    if env is None:
        return GpuEnv()
    return GpuEnv(**env)


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
