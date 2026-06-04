from logging import captureWarnings
from sys import version_info

import hydra
from packaging.version import Version

if version_info >= (3, 14) and Version(hydra.__version__) <= Version('1.3.2'):
    from argparse import ArgumentParser

    ArgumentParser._check_help = lambda _self, _action: None  # ty: ignore

captureWarnings(capture=True)
