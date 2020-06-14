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
    val = val / mag[:, None]
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
    spc = val.mean(axis=1)
    tf.summary.histogram(f'peak/{stage:03d}', mag, step=step)
    tf.summary.histogram(f'sparseness/{stage:03d}', spc, step=step)
    tf.summary.image(f'cor/{stage:03d}', _jet(cor)[None, ...], step=step)
    tf.summary.image(f'spike/{stage:03d}', _reds(val)[None, ...], step=step)


def summary_footprint_stat(val, mask, stage, step=0):
    cor = calc_cor(val)

    nk, nx = val.shape
    h, w = mask.shape

    imgs_max = np.zeros((1, h, w))
    imgs_max[0, mask] = val.max(axis=0)
    tf.summary.image(f'cor/{stage:03d}', _jet(cor)[None, ...], step=step)
    tf.summary.image(f'max/{stage:03d}', _greens(imgs_max), step=step)


def summary_footprint_sample(val, mask, stage, step):
    h, w = mask.shape
    imgs = np.zeros((4, h, w))
    imgs[:, mask] = val[[0, 1, -2, -1]]
    tf.summary.image(
        f'footprint-sample/{stage:03d}',
        _greens(imgs), max_outputs=4, step=step,
    )


def summary_footprint(val, mask, stage):
    h, w = mask.shape
    img = np.zeros((h, w))
    for i, v in enumerate(val[::-1]):
        img[:] = 0.0
        img[mask] = v
        tf.summary.image(
            f'footprint/{stage:03d}', _greens(img)[None, ...], step=i+1,
        )


class HotaruCallback(tf.keras.callbacks.TensorBoard):

    def __init__(self, stage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage = stage

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        mode = K.get_value(self.model.variance._mode)
        stage = self.stage
        writer = self._get_writer(self._train_run_name)
        with writer.as_default():
            if mode == 0:
                summary_spike(self.model.spike.val, stage, step=epoch)
            else:
                val, mag = normalized_and_sort(self.model.footprint.val)
                spc = val.mean(axis=1)
                mask = self.model.mask
                tf.summary.histogram(f'peak/{stage:03d}', mag, step=epoch)
                tf.summary.histogram(
                    f'sparseness/{stage:03d}', spc, step=epoch,
                )
                summary_footprint_stat(val, mask, stage, step=epoch)
                summary_footprint_sample(val, mask, stage, step=epoch)
