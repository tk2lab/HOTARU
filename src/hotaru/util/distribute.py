from enum import Enum

import tensorflow as tf


class MirroredStrategy(tf.distribute.MirroredStrategy):
    def close(self):
        self._extended._collective_ops._pool.close()


class ReduceOp(Enum):
    CONCAT = 0
    STACK = 1
    SUM = 2
    MIN = 3
    MAX = 4


def distributed(*types, strategy=None, loop=True):
    def make_init(t):
        if t == ReduceOp.SUM:
            return tf.constant(0, tf.float32)
        elif t == ReduceOp.MIN:
            return ()
        elif t == ReduceOp.MAX:
            return ()
        elif t == ReduceOp.CONCAT:
            return ()
        elif t == ReduceOp.STACK:
            return ()
        else:
            raise NotImplementedError()

    def serialize(x, t):
        if t == ReduceOp.SUM:
            return strategy.reduce(tf.distribute.ReduceOp.SUM, x, axis=None)
        elif t == ReduceOp.MIN:
            return strategy.experimental_local_results(x)
        elif t == ReduceOp.MAX:
            return strategy.experimental_local_results(x)
        elif t == ReduceOp.CONCAT:
            return strategy.experimental_local_results(x)
        elif t == ReduceOp.STACK:
            return strategy.experimental_local_results(x)
        else:
            raise NotImplementedError()

    def finish(o, t):
        if t == ReduceOp.SUM:
            return o
        elif t == ReduceOp.MIN:
            return tf.reduce_min(o, axis=0)
        elif t == ReduceOp.MAX:
            return tf.reduce_max(o, axis=0)
        elif t == ReduceOp.CONCAT:
            return tf.concat(o, axis=0)
        elif t == ReduceOp.STACK:
            return tf.stack(o)
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
                        dx = d[0]
                    else:
                        dx = d
                    if isinstance(dx, tf.distribute.DistributedValues):
                        for di in dx.values:
                            prog.update(tf.shape(di)[0].numpy())
                    else:
                        prog.update(tf.shape(dx)[0].numpy())
            out = tuple(finish(o, t) for o, t in zip(os, types))
            return out

        return loop_run

    if strategy is None:
        strategy = tf.distribute.get_strategy()
    return decorator
