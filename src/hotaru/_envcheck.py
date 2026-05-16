from importlib.util import find_spec
from logging import getLogger
from os import environ

__all__ = []

pkg = __package__ or 'hotaru'
logger = getLogger(pkg)

backend = environ.get('KERAS_BACKEND')

if backend is None:
    for module in ('jax', 'torch'):
        if find_spec(module) is not None:
            backend = module
            break

if backend is None:
    raise ImportError(
        'No backend detected. Please install one of the supported backends: '
        f"pip install '{pkg}[jax]' or '{pkg}[torch]' (experimental)"
    )

environ['KERAS_BACKEND'] = backend
logger.info(f'use {backend} backend')
