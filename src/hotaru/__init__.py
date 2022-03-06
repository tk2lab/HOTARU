import pkg_resources


__all__ = [
    '__version__',
]


__version__ = pkg_resources.get_distribution('hotaru').version
