from importlib import metadata

from . import _envcheck

__version__ = metadata.version(__package__ or 'hotaru')
