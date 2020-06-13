import tensorflow.keras.backend as K


def neighbor(imgs):
    fil = K.constant([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8.0
    fil = K.reshape(fil, [3, 3, 1, 1])
    return K.conv2d(
        imgs[..., None], fil, [1, 1, 1, 1], padding='SAME',
    )[..., 0]
