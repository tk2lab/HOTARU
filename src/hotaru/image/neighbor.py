import tensorflow as tf


def neighbor(imgs):
    imgs = imgs[..., None]
    fil = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    fil = tf.convert_to_tensor(fil, tf.float32) / 8.0
    fil = tf.reshape(fil, [3, 3, 1, 1])
    return tf.nn.conv2d(imgs, fil, [1, 1, 1, 1], padding='SAME')[..., 0]
