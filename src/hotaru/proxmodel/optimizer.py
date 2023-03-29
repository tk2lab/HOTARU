import tensorflow as tf

TF_VERSION = [int(v) for v in tf.__version__.split(".")]
if TF_VERSION[0] != 2:
    raise RuntimeError()
if TF_VERSION[1] >= 11:
    from .optimizer_experimental import ProxOptimizer
else:
    from .optimizer_regacy import ProxOptimizer
