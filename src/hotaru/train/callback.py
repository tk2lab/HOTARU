import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
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


def spike_summary(val, step=0):
    val, mag = normalized_and_sort(val)
    cor = calc_cor(val)
    tf.summary.histogram('magnitude/spike', mag, step=step)
    tf.summary.image('spike-val', _reds(val)[None, ...], step=step)
    tf.summary.image('cor/spike', _jet(cor)[None, ...], step=step)


def footprint_summary(val, mask, mag=None, name='footprint', step=0):
    if mag is None:
        val, mag = normalized_and_sort(val)
    else:
        idx = np.argsort(mag)[::-1]
        val = val[idx]
    cor = calc_cor(val)

    nk, nx = val.shape
    h, w = mask.shape
    imgs = np.zeros((nk, h, w))
    imgs[:, mask] = val

    imgs_max = imgs.max(axis=0)
    tf.summary.histogram(f'area/{name}', val.sum(axis=1), step=step)
    tf.summary.image(f'cor/{name}', _jet(cor)[None, ...], step=step)
    tf.summary.image(f'max/{name}', _greens(imgs_max)[None, ...], step=step)
    if name == 'footprint':
        tf.summary.histogram(f'magnitude/{name}', mag, step=step)
        tf.summary.image(f'{name}-val', _greens(imgs[[0,1,-2,-1]]), max_outputs=4, step=step)
    else:
        tf.summary.histogram(name, mag, step=step)
        for i, img in enumerate(imgs):
            tf.summary.image(f'{name}-val', _greens(img[None, ...]), step=i)


class HotaruCallback(tf.keras.callbacks.TensorBoard):

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        mode = K.get_value(self.model.variance._mode)
        writer = self._get_writer(self._train_run_name)
        with writer.as_default():
            if mode == 0:
                spike_summary(self.model.spike.val, step=epoch)
            else:
                mask = self.model.mask
                footprint_summary(self.model.footprint.val, mask, step=epoch)
