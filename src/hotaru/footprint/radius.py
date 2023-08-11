import jax.numpy as jnp


def get_radius(cfg):
    if isinstance(cfg, tuple):
        return cfg
    match cfg["type"]:
        case "logscale":
            r = jnp.geomspace(cfg["min"], cfg["max"], cfg["num"])
        case "linear":
            r = jnp.linspace(cfg["min"], cfg["max"], cfg["num"])
        case "list":
            r = jnp.array(cfg["val"])
    return tuple(float(ri) for ri in r)
