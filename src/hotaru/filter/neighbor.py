import tensorflow as tf


def neighbor(imgs):
    fil = tf.constant([[1, 1, 1], [1, 0, 1], [1, 1, 1]], tf.float32) / 8
    fil = tf.reshape(fil, [3, 3, 1, 1])
    return tf.nn.conv2d(
        imgs[..., None],
        fil,
        [1, 1, 1, 1],
        padding="SAME",
    )[..., 0]
