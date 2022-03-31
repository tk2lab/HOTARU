from functools import wraps

import click

from hotaru.util.timer import Timer


def run_base(func):

    @wraps(func)
    @click.pass_obj
    def new_func(obj, **args):
        if not obj.need_exec(func.__name__):
            return

        with Timer() as timer:
            log = func(obj, **args)

        log.update(args)
        log['time'] = timer.get()
        obj.save_log(func.__name__, log)

    return new_func
