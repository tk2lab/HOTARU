from cleo import Application as ApplicationBase

from ..version import __version__

from .config import ConfigCommand
from .update import UpdateCommand
from .run import RunCommand

from .data import DataCommand
from .find import FindCommand
from .init import InitCommand
from .temporal import TemporalCommand
from .spatial import SpatialCommand
from .clean import CleanCommand

from .output import OutputCommand
from .fig.mpeg import MpegCommand
from .fig.data import FigDataCommand 
from .fig.find import FigFindCommand 
from .fig.init import FigInitCommand
from .fig.temporal import FigTemporalCommand
from .fig.spatial import FigSpatialCommand
from .fig.clean import FigCleanCommand


class Application(ApplicationBase):

    def __init__(self):
        super().__init__('hotaru', __version__)

        self.add_commands(
            ConfigCommand(),
            UpdateCommand(),
            RunCommand(),
            OutputCommand(),
            MpegCommand(),

            DataCommand(),
            FindCommand(),
            InitCommand(),
            TemporalCommand(),
            SpatialCommand(),
            CleanCommand(),

            FigDataCommand(),
            FigFindCommand(),
            FigInitCommand(),
            FigTemporalCommand(),
            FigSpatialCommand(),
            FigCleanCommand(),
        )
