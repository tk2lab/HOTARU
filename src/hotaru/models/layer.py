from keras.layers import Layer as KerasLayer

from ..saving import Serializable


class Layer(Serializable, KerasLayer):
    pass
