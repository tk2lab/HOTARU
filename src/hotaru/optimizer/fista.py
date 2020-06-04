# -*- coding: utf-8 -*-

import tensorflow as tf

from ..util.module import Module


class FastProxOptimizer(Module):

    def __init__(self, model=None, xs=None, steps=100, scale=20.0, fac_lr=0.7, name='FISTA'):
        super().__init__(name=name)
        self.model = model
        self.add_variable('steps', (), tf.int32, init=steps)
        self.add_variable('scale', (), init=scale)
        self.add_variable('fac_lr', (), init=fac_lr)

        self.add_variable('step', ())
        self.add_variable('lr', ())
        self.xs = xs or []
        with self.name_scope:
            self.zs = [tf.Variable(x, shape=x.shape, name='zs') for x in self.xs]

    def score(self):
        return self.model.loss(self.xs) + self.model.penalty(self.xs)

    def train(self):
        self.step.assign(0.0)
        for z, x in zip(self.zs, self.xs):
            z.assign(x)
        score = [self.score().numpy()]
        for step in tf.range(self.steps):
            score.append(self.minimize().numpy())
        return score

    @tf.function(input_signature=[])
    def minimize(self):

        def loop(lr):
            ds = [y - lr * g for y, g in zip(ys, gs)]
            ns = self.model.prox(ds, lr)
            loss_new = self.model.loss(ns)
            return ns, loss_new

        def est_loss(lr):
            loss_est = loss
            for y, g, n in zip(ys, gs, ns):
                d = n - y
                loss_est += tf.reduce_sum(d * g)
                loss_est += (0.5 / lr) * tf.reduce_sum(d * d)
            return loss_est

        t = self.scale / (self.step + self.scale)
        ys = [(1.0 - t) * x + t * z for x, z in zip(self.xs, self.zs)]
        with tf.GradientTape() as tape:
            for y in ys:
                tape.watch(y)
            loss = self.model.loss(ys)
        gs = tape.gradient(loss, ys)

        lr = self.lr.read_value()
        ns, loss_new = loop(lr)
        if self.fac_lr < 1.0:
            loss_est = est_loss(lr)
            while tf.math.is_nan(loss_new) | (loss_new > loss_est):
                #tf.print('  retry', self.step, loss_new, loss_est, lr)
                lr *= self.fac_lr
                ns, loss_new = loop(lr)
                loss_est = est_loss(lr)

        self.lr.assign(lr)
        self.step.assign_add(1.0)
        for z, x, n in zip(self.zs, self.xs, ns):
            z.assign((1.0 - 1/t) * x + (1/t) * n)
            x.assign(n)

        return loss_new + self.model.penalty(ns)
