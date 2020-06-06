import os

import tensorflow as tf

from cleo import Application as ApplicationBase
#from cleo.config import ApplicationConfig

from ..version import __version__
from .config import ConfigCommand
from .run import RunCommand
from .status import StatusCommand
from .data import DataCommand
from .peak import PeakCommand
from .make import MakeCommand
from .spike import SpikeCommand
from .footprint import FootprintCommand
from .clean import CleanCommand


class Application(ApplicationBase):

    def __init__(self):
        super().__init__('hotaru', __version__)

        self.job_dir = None
        self.current_key = dict(peak=None, footprint=None, spike=None)
        self.current_val = dict(peak=None, footprint=None, spike=None)

        self.add_commands(
            ConfigCommand(),
            RunCommand(),
            StatusCommand(),
            DataCommand(),
            PeakCommand(),
            MakeCommand(),
            SpikeCommand(),
            FootprintCommand(),
            CleanCommand(),
        )
