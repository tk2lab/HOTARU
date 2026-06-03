from importlib import metadata

from . import _envcheck
from . import _hydra_patch

__version__ = metadata.version(__package__ or 'hotaru')
