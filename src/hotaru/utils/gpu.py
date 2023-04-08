import subprocess
import os


def get_gpu_memory():
    result = subprocess.run([
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ], stdout=subprocess.PIPE, encoding="utf-8", check=True)
    return [int(x) for x in result.stdout.strip().split(os.linesep)]
