from logging import getLogger

import numpy as np
from keras.utils import PyDataset

from ..typing import Array
from .imgs import MovieData

logger = getLogger(__name__)


class MovieDataset(PyDataset):
    def __init__(self, data: MovieData, batch_size: int = -1, **kwargs):
        super().__init__(**kwargs)
        if batch_size == -1:
            batch_size = data.num_frames
        self.ts = np.arange(data.num_frames, dtype='int32')
        self.data = data
        self.batch_size = batch_size

    def on_epoch_end(self):
        pass

    def __len__(self) -> int:
        return (self.ts.size + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int) -> tuple[Array, Array]:
        s = self.batch_size * index
        e = s + self.batch_size
        diff = max(e - self.ts.size, 0)
        ts, imgs = self.ts[s:e], self.data.imgs[s:e]
        ts = np.pad(ts, ((0, diff),), constant_values=-1)
        imgs = np.pad(imgs, ((0, diff), (0, 0), (0, 0)))
        return ts, imgs
