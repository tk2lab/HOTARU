import tensorflow as tf
from tensorflow.python.saved_model.revived_types import VersionedTypeRegistration
from tensorflow.python.saved_model.revived_types import register_revived_type


class Module(tf.Module):

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        def predicate(obj):
            return type(obj) is cls

        def factory(proto):
            return cls()

        version = VersionedTypeRegistration(factory, 1,1,1)
        register_revived_type(cls.__name__, predicate, [version])

    def add_variable(self, name, shape, dtype=tf.float32, init=None):
        real_shape = (0 if d is None else d for d in shape)
        init = init or tf.zeros(real_shape, dtype)
        with self.name_scope:
            v = tf.Variable(init, shape=tf.TensorShape(shape), name=name)
        setattr(self, name, v)

    def add_module(self, name, mod):
        with self.name_scope:
            setattr(self, name, mod)
