import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from matplotlib import cm


class HotaruCallback(tf.keras.callbacks.TensorBoard):

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        jet = cm.get_cmap('jet')
        reds = cm.get_cmap('Reds')
        greens = cm.get_cmap('Greens')
        mode = K.get_value(self.model.variance._mode)
        writer = self._get_writer(self._train_run_name)
        with writer.as_default():
            if mode == 0:
                val = self.model.spike.val
                m = val.max(axis=1)
                tf.summary.histogram('spike/magnitude', m, step=epoch)
                mag = val.max(axis=1)
                idx = np.argsort(mag)[::-1]
                val /= mag[:, None]
                val = val[idx]
                cor = val @ val.T
                scale = np.sqrt(np.diag(cor))
                cor /= scale * scale[:, None]
                tf.summary.image('spike-val', reds(val)[None, ...], step=epoch)
                tf.summary.image('cor/spike', jet(cor)[None, ...], step=epoch)
            else:
                val = self.model.footprint.val
                mag = val.max(axis=1)
                idx = np.argsort(mag)[::-1]
                val /= mag[:, None]
                val = val[idx]
                cor = val @ val.T
                scale = np.sqrt(np.diag(cor))
                cor /= scale * scale[:, None]
                mask = self.model.mask
                nk, nx = val.shape
                h, w = mask.shape
                imgs = np.zeros((nk, h, w))
                imgs[:, mask] = val
                tf.summary.histogram('footprint/magnitude', mag, step=epoch)
                tf.summary.histogram('footprint/area', val.sum(axis=1), step=epoch)
                tf.summary.image('cor/footprint', jet(cor)[None, ...], step=epoch)
                tf.summary.image('footprint-max', greens(imgs.max(axis=0, keepdims=True)), step=epoch)
                tf.summary.image('footprint-val/0', greens(imgs[:3]), step=epoch)
                tf.summary.image('footprint-val/1', greens(imgs[-3:]), step=epoch)
