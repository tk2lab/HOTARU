from enum import Enum

import tensorflow as tf


class ReduceOp(Enum):
    CONCAT = 0
    STACK = 1
    SUM = 2
    MIN = 3
    MAX = 4
    LIST = 5


def distributed(*types):
    """"""

    def decorator(func):
        """"""

        def loop_run(data, *args, **kwargs):
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
                elif t == ReduceOp.LIST:
                    return ()
                else:
                    raise NotImplementedError()

            def serialize(x, t):
                if t == ReduceOp.SUM:
                    return strategy.reduce(
                        tf.distribute.ReduceOp.SUM, x, axis=None
                    )
                elif t == ReduceOp.MIN:
                    return strategy.experimental_local_results(x)
                elif t == ReduceOp.MAX:
                    return strategy.experimental_local_results(x)
                elif t == ReduceOp.CONCAT:
                    return strategy.experimental_local_results(x)
                elif t == ReduceOp.STACK:
                    return strategy.experimental_local_results(x)
                elif t == ReduceOp.LIST:
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
                elif t == ReduceOp.LIST:
                    return sum([tf.unstack(oi) for oi in o], [])
                else:
                    raise NotImplementedError()

            def dist_run(*args, **kwargs):
                xs = strategy.run(func, args, kwargs)
                if len(types) == 1:
                    xs = (xs,)
                return tuple(serialize(x, t) for x, t in zip(xs, types))

            strategy = tf.distribute.get_strategy()
            os = tuple(make_init(t) for t in types)
            for d in strategy.experimental_distribute_dataset(data):
                xs = dist_run(d, *args, **kwargs)
                os = tuple(o + x for o, x in zip(os, xs))
            out = tuple(finish(o, t) for o, t in zip(os, types))
            if len(types) == 1:
                out = out[0]
            return out

        return loop_run

    return decorator
