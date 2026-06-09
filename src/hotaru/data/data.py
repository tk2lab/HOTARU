from logging import getLogger

from ..models import Layer
from ..saving import Config
from ..typing import Shape
from .dataset import CalciumImagingDataset
from .io import apply_mask
from .io import load_imgs

logger = getLogger(__name__)


class CalciumImagingData(Layer):
    def __init__(self, **kwargs):
        imgs_kwargs = kwargs.pop('imgs', {})
        mask_kwargs = kwargs.pop('mask', {'kind': 'nomask'})

        if 'path' in imgs_kwargs:
            if 'path' in kwargs:
                raise ValueError()
            path = imgs_kwargs.pop('path')
        else:
            if 'path' not in kwargs:
                raise ValueError()
            path = kwargs.pop('path')
        if 'hz' in imgs_kwargs:
            if 'hz' in kwargs:
                raise ValueError()
            hz = imgs_kwargs.pop('hz')
        else:
            if 'hz' not in kwargs:
                raise ValueError()
            hz = kwargs.pop('hz')

        super().__init__(**kwargs)
        self.imgs_kwargs = imgs_kwargs
        self.mask_kwargs = mask_kwargs

        imgs = load_imgs(path=path, hz=hz, **imgs_kwargs)
        imgs, mask = apply_mask(imgs, **mask_kwargs)
        self.imgs = imgs
        self.mask = mask
        self.hz = hz
        self.imgs_path = path

    def get_config(self) -> Config:
        return {
            'path': str(self.imgs_path),
            'hz': self.hz,
            'imgs': self.imgs_kwargs,
            'mask': self.mask_kwargs,
            **super().get_config(),
        }

    @property
    def shape(self) -> Shape:
        return self.imgs.shape

    @property
    def num_frames(self) -> int:
        return self.imgs.shape[0]

    @property
    def width(self) -> int:
        return self.imgs.shape[2]

    @property
    def height(self) -> int:
        return self.imgs.shape[1]

    def dataset(self, *args, **kwargs) -> CalciumImagingDataset:
        return CalciumImagingDataset(self.imgs, *args, **kwargs)
