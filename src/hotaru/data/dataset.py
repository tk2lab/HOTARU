from logging import getLogger

from keras import ops
from keras.utils import PyDataset

from ..typing import Array
from ..typing import Tensor

logger = getLogger(__name__)


class CalciumImagingDataset(PyDataset):
    def __init__(self, imgs: Array, batch_size: int = -1, **kwargs):
        super().__init__(**kwargs)
        nt = imgs.shape[0]
        if batch_size == -1:
            batch_size = nt
        self.ts = ops.arange(nt, dtype='int32')
        self.imgs = imgs
        self.batch_size = batch_size

    def on_epoch_end(self):
        pass

    def __len__(self) -> int:
        return (self.ts.size + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        s = self.batch_size * index
        e = s + self.batch_size
        ts, imgs = self.ts[s:e], self.imgs[s:e]

        diff = self.batch_size - ts.size
        ts = ops.pad(ts, ((0, diff),), constant_values=-1)
        imgs = ops.pad(imgs, ((0, diff), (0, 0), (0, 0)))

        return ts, imgs
