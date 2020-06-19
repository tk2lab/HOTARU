from enum import Enum

import tensorflow as tf


class ReduceOp(Enum):
    CONCAT = 0
    SUM = 1
    MAX = 2


def distributed(*types, loop=True):

    def make_init(t):
        if t == ReduceOp.SUM:
            return tf.constant(0, tf.float32)
        elif t == ReduceOp.MAX:
            return ()
        elif t == ReduceOp.CONCAT:
            return ()
        else:
            raise NotImplementedError()

    def serialize(x, t):
        if t == ReduceOp.SUM:
            return strategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=None)
        elif t == ReduceOp.MAX:
            return strategy.experimental_local_results(x)
        elif t == ReduceOp.CONCAT:
            return strategy.experimental_local_results(x)
        else:
            raise NotImplementedError()

    def finish(o, t):
        if t == ReduceOp.SUM:
            return o
        elif t == ReduceOp.MAX:
            return tf.reduce_max(o, axis=0)
        elif t == ReduceOp.CONCAT:
            return tf.concat(o, axis=0)
        else:
            raise NotImplementedError()

    def decorator(func):

        @tf.function(experimental_relax_shapes=True)
        def dist_run(*args, **kwargs):
            xs = strategy.run(func, args, kwargs)
            return tuple(serialize(x, t) for x, t in zip(xs, types))

        def loop_run(data, *args, prog=None, **kwargs):
            os = tuple(make_init(t) for t in types)
            for d in strategy.experimental_distribute_dataset(data):
                xs = dist_run(d, *args, **kwargs)
                os = tuple(o + x for o, x in zip(os, xs))
                if prog is not None:
                    if isinstance(d, tuple):
                        prog.add(tf.shape(d[0])[0].numpy())
                    else:
                        prog.add(tf.shape(d)[0].numpy())
            return tuple(finish(o, t) for o, t in zip(os, types))

        def single_run(*args, **kwargs):
            xs = dist_run(*args, **kwargs)
            return tuple(finish(x, t) for x, t in zip(xs, types))

        return loop_run if loop else single_run

    strategy = tf.distribute.get_strategy()
    return decorator
