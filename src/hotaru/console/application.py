import os

import tensorflow as tf
from cleo import Application as ApplicationBase
#from cleo.config import ApplicationConfig

from ..version import __version__
from .config import ConfigCommand
from .run import RunCommand
from .status import StatusCommand
from .history import HistoryCommand
from .data import DataCommand
from .peak import PeakCommand
from .segment import SegmentCommand
from .spike import SpikeCommand
from .footprint import FootprintCommand
from .clean import CleanCommand


class Application(ApplicationBase):

    def __init__(self):
        super().__init__('hotaru', __version__)

        self.job_dir = None
        self.current_key = dict(
            peak=None, footprint=None, spike=None, clean=None,
        )
        self.current_val = dict(
            peak=None, footprint=None, spike=None, clean=None,
        )

        self.add_commands(
            ConfigCommand(),
            RunCommand(),
            StatusCommand(),
            HistoryCommand(),
            DataCommand(),
            PeakCommand(),
            SegmentCommand(),
            SpikeCommand(),
            FootprintCommand(),
            CleanCommand(),
        )
