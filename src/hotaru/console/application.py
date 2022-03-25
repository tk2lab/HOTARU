from cleo import Application as ApplicationBase

from ..version import __version__

from .data import DataCommand
from .find import FindCommand
from .reduce import ReduceCommand
from .init import InitCommand
from .temporal import TemporalCommand
from .spatial import SpatialCommand
from .clean import CleanCommand

from .config import ConfigCommand
from .update import UpdateCommand
from .run import RunCommand

#from .output import OutputCommand
#from .history import HistoryCommand
#from .test import TestCommand


class Application(ApplicationBase):

    def __init__(self):
        super().__init__('hotaru', __version__)

        self.add_commands(
            DataCommand(),
            FindCommand(),
            ReduceCommand(),
            InitCommand(),
            TemporalCommand(),
            SpatialCommand(),
            CleanCommand(),

            ConfigCommand(),
            UpdateCommand(),
            RunCommand(),

            #OutputCommand(),
            #HistoryCommand(),
            #TestCommand(),
        )
