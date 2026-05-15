from importlib import metadata

from . import _envcheck
from .models import Model

__version__ = metadata.version(__package__ or 'hotaru')
