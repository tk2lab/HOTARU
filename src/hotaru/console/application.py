from cleo import Application as ApplicationBase
import tensorflow as tf

from ..version import __version__
from .config import ConfigCommand
from .run import RunCommand
from .history import HistoryCommand
from .data import DataCommand
from .peak import PeakCommand
from .segment import SegmentCommand
from .spike import SpikeCommand
from .footprint import FootprintCommand
from .clean import CleanCommand
from .output import OutputCommand
from .test import TestCommand


class Application(ApplicationBase):

    def __init__(self):
        super().__init__('hotaru', __version__)

        self.job_dir = None
        self.strategy = tf.distribute.MirroredStrategy()

        self.add_commands(
            ConfigCommand(),
            RunCommand(),
            HistoryCommand(),
            DataCommand(),
            PeakCommand(),
            SegmentCommand(),
            SpikeCommand(),
            FootprintCommand(),
            CleanCommand(),
            OutputCommand(),
            TestCommand(),
        )
