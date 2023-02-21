from ..train.model import HotaruModel
from .extend import HotaruExtendMixin
from .output import HotaruOutputMixin


class Hotaru(HotaruModel, HotaruExtendMixin, HotaruOutputMixin):
    pass
