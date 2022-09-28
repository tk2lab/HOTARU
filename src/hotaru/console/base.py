import functools

import click
import numpy as np

from ..util.timer import Timer

readable_file = click.Path(exists=True, dir_okay=False, readable=True)


def configure(ctx, param, tag):
    if tag is None:
        tag = ctx.obj.config.get("main", param.name)
    ctx.default_map = ctx.obj.config.defaults()
    for section in [ctx.info_name, f"{ctx.info_name}/{tag}"]:
        if section in ctx.obj.config:
            ctx.default_map.update(ctx.obj.config[section])
    return tag


def command_wrap(command):
    """"""

    @functools.wraps(command)
    def wrapped_command(obj, tag, **args):
        kind = command.__name__

        if obj.can_skip(tag, kind, **args):
            click.echo("    skip")
            return

        for k, v in args.items():
            click.echo(f"   {k}: {v}")

        with Timer() as timer:
            log = command(obj, tag, **args)
        time = timer.get()
        click.echo(f"time: {time[0]}")

        if log:
            log, tag, stage, kind = log
            args.update(log)
            args["time"] = time
            obj.save_log(args, tag, stage, kind)
    return wrapped_command


def wrap_option(prefix, **kwargs):
    wrap = kwargs.setdefault(prefix, {})
    for k in list(kwargs.keys()):
        if k.startswith(f"{prefix}_"):
            wrap[k[len(prefix)+1:]] = kwargs.pop(k)
    return kwargs


def get_radius(type, min, max, num):
    if type == "linear":
        func = np.linspace
    elif type == "log":
        func = np.logspace
        min = np.log10(min)
        max = np.log10(max)
    return func(min, max, num)


def radius_options(command):
    @click.option("--radius-type", type=click.Choice(["log", "linear"]))
    @click.option("--radius-min", type=float)
    @click.option("--radius-max", type=float)
    @click.option("--radius-num", type=int)
    @functools.wraps(command)
    def wrapped_command(*args, **kwargs):
        kwargs = wrap_option("radius", **kwargs)
        kwargs["radius"] = get_radius(**kwargs["radius"])
        return command(*args, **kwargs)
    return wrapped_command


def dynamics_options(command):
    @click.option("--dynamics-tau1", type=float)
    @click.option("--dynamics-tau2", type=float)
    @functools.wraps(command)
    def wrapped_command(*args, **kwargs):
        return command(*args, **wrap_option("dynamics", **kwargs))
    return wrapped_command


def penalty_options(command):
    @click.option("--penalty-footprint", type=float)
    @click.option("--penalty-spike", type=float)
    @click.option("--penalty-localx", type=float)
    @click.option("--penalty-localt", type=float)
    @click.option("--penalty-spatial", type=float)
    @click.option("--penalty-temporal", type=float)
    @functools.wraps(command)
    def wrapped_command(*args, **kwargs):
        return command(*args, **wrap_option("penalty", **kwargs))
    return wrapped_command


def optimizer_options(command):
    @click.option("--optimizer-learning-rate", type=float)
    @click.option("--optimizer-nesterov-scale", type=float)
    @click.option("--optimizer-reset-interval", type=int)
    @functools.wraps(command)
    def wrapped_command(*args, **kwargs):
        return command(*args, **wrap_option("optimizer", **kwargs))
    return wrapped_command


def early_stop_options(command):
    @click.option("--early-stop-min-delta", type=float)
    @click.option("--early-stop-patience", type=int)
    @functools.wraps(command)
    def wrapped_command(*args, **kwargs):
        return command(*args, **wrap_option("early_stop", **kwargs))
    return wrapped_command


def threshold_options(command):
    @click.option("--threshold-sim", type=click.FloatRange(0.0, 1.0))
    @click.option("--threshold-area", type=click.FloatRange(0.0, 1.0))
    @click.option("--threshold-overwrap", type=click.FloatRange(0.0, 1.0))
    @functools.wraps(command)
    def wrapped_command(*args, **kwargs):
        return command(*args, **wrap_option("threshold", **kwargs))
    return wrapped_command
