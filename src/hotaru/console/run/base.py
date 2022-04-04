from functools import wraps

import click

from hotaru.util.timer import Timer

from ..obj import Obj


def run_base(func):

    @wraps(func)
    @click.pass_obj
    def new_func(obj, **args):
        obj['kind'] = func.__name__

        sections = [obj.kind, f'{obj.kind}.{obj.tag}']
        stage = args.get('stage')
        if stage is not None:
            sections.append(f'{obj.kind}.{obj.tag}.{stage}')
        for sec in sections:
            if obj.config.has_section(sec):
                for key in args.keys():
                    if obj.config.has_option(sec, key):
                        args[key] = cfg.get(sec, key)
        obj.update(args)

        if not obj.need_exec():
            return

        with Timer() as timer:
            log = func(obj)
        click.echo(f'{timer.get()[0]} sec')

        if log is not None:
            log['time'] = timer.get()
            log.update(args)
            obj.save_log(obj.kind, log)

    return new_func
