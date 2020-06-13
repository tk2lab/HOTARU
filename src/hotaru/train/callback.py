import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from matplotlib import cm


_jet = cm.get_cmap('jet')
_reds = cm.get_cmap('Reds')
_greens = cm.get_cmap('Greens')


def normalized_and_sort(val):
    mag = val.max(axis=1)
    idx = np.argsort(mag)[::-1]
    val /= mag[:, None]
    val = val[idx]
    return val, mag


def calc_cor(val):
    cor = val @ val.T
    scale = np.sqrt(np.diag(cor))
    cor /= scale * scale[:, None]
    return cor


def summary_spike(val, stage, step=0):
    val, mag = normalized_and_sort(val)
    cor = calc_cor(val)
    tf.summary.histogram('magnitude{stage}', mag, step=step)
    tf.summary.image('cor{stage}', _jet(cor)[None, ...], step=step)
    tf.summary.image('spike{stage}', _reds(val)[None, ...], step=step)


def summary_footprint_stat(val, mask, n, step=0):
    cor = calc_cor(val)

    nk, nx = val.shape
    h, w = mask.shape

    imgs_max = np.zeros((h, w))
    imgs_max[mask] = val.max(axis=0)
    tf.summary.histogram(f'area{n}', val.sum(axis=1), step=step)
    tf.summary.image(f'cor{n}', _jet(cor)[None, ...], step=step)
    tf.summary.image(f'max{n}', _greens(imgs_max)[None, ...], step=step)


def summary_footprint_sample(val, mask, n, step):
    h, w = mask.shape
    imgs = np.zeros((4, h, w))
    imgs[:, mask] = val[[0, 1, -2, -1]]
    tf.summary.image(
        f'footprint-sample{n}', _greens(imgs), max_outputs=4, step=step,
    )


def summary_footprint(val, mask, n):
    h, w = mask.shape
    img = np.zeros((h, w))
    for i, v in enumerate(val):
        img[:] = 0.0
        img[mask] = v
        tf.summary.image(
            f'footprint{n}', _greens(img)[None, ...], step=i,
        )


class HotaruCallback(tf.keras.callbacks.TensorBoard):

    def __init__(self, stage='000', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        mode = K.get_value(self.model.variance._mode)
        writer = self._get_writer(self._train_run_name)
        with writer.as_default():
            if mode == 0:
                summary_spike(self.model.spike.val, self.stage, step=epoch)
            else:
                mask = self.model.mask
                summary_footprint_stat(
                    self.model.footprint.val, mask, self.stage, step=epoch,
                )
                summary_footprint_sample(
                    self.model.footprint.val, mask, self.stage, step=epoch,
                )
