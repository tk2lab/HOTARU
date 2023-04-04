import jax.numpy as jnp


class SaverMixin:

    def save(self, path):
        jnp.savez(path, **self._asdict())

    @classmethod
    def load(cls, path):
        with jnp.load(path) as npz:
            stats = cls(*[npz[n] for n in cls._fields])
        return stats
