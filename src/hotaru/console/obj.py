import os

import click
import numpy as np
import pandas as pd
import tensorflow as tf

from ..io.csv import load_csv
from ..io.csv import save_csv
from ..io.pickle import load_pickle
from ..io.pickle import save_pickle
from ..io.numpy import load_numpy
from ..io.numpy import save_numpy
from ..io.tfrecord import load_tfrecord
from ..io.tfrecord import save_tfrecord
from ..io.tiff import save_tiff
from ..train.model import HotaruModel as Model
from .progress import ProgressCallback


class Obj:
    """HOTARU CLI Object."""

    def __init__(self):
        self._log = {}
        self._model_tag = None

    def __enter__(self):
        self.strategy = tf.distribute.MirroredStrategy()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if int(tf.version.VERSION.split(".")[1]) <= 9:
            self.strategy._extended._collective_ops._pool.close()

    def update(self, x):
        for k, v in x.items():
            setattr(self, k, v)

    def invoke(self, ctx, command, *args):
        ctx = command.make_context(command.name, list(args), ctx)
        command.invoke(ctx)

    def get_config(self, kind, tag, key):
        if f"{kind}/{tag}" in self.config:
            return self.config.get(f"{kind}/{tag}", key)
        else:
            return self.config.get(kind, key)

    # Model

    def model(self, data_tag, nk=1):
        if self._model_tag != data_tag:
            prev_log = self.log(data_tag, 0, "1data")
            data = self.data(data_tag)
            nx = prev_log["nx"]
            nt = prev_log["nt"]
            hz = prev_log["hz"]
            tausize = prev_log["tausize"]

            model = Model(name="Hotaru")
            model.build(data, nk, nx, nt, hz, tausize, self.strategy)
            model.compile_temporal()
            model.compile_spatial()
            self._model_tag = data_tag
            self._model = model
        return self._model

    def callbacks(self, label, log_dir):
        return [
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                update_freq="batch",
                write_graph=False,
            ),
            ProgressCallback(label),
        ]

    # SUMMARY

    def summary_path(self, tag, stage, kind):
        return f"{self.workdir}/summary/{tag}/{stage:03}_{kind}"

    # SAVE

    def out_path(self, tag, stage, kind, name):
        path = f"{self.workdir}/{tag}/{stage:03}_{kind}"
        os.makedirs(path, exist_ok=True)
        return f"{path}/{name}"

    def save_tfrecord(self, data, *args, **kwargs):
        out_path = self.out_path(*args, **kwargs)
        save_tfrecord(f"{out_path}.tfrecord", data)

    def save_numpy(self, data, *args, **kwargs):
        out_path = self.out_path(*args, **kwargs)
        save_numpy(f"{out_path}.npy", data)

    def save_csv(self, data, *args, **kwargs):
        out_path = self.out_path(*args, **kwargs)
        save_csv(f"{out_path}.csv", data)

    def save_tiff(self, data, *args, **kwargs):
        out_path = self.out_path(*args, **kwargs)
        save_tiff(f"{out_path}.tif", data)

    # LOG

    def save_log(self, log, *args, **kwargs):
        path = self.out_path(*args, **kwargs, name="log.pickle")
        save_pickle(path, log)
        self._log[path] = log

    def log(self, *args, **kwargs):
        path = self.out_path(*args, **kwargs, name="log.pickle")
        if path not in self._log:
            self._log[path] = load_pickle(path)
        return self._log[path]

    def can_skip(self, tag, kind, **args):
        if kind == "make":
            stage = 1
        elif kind == "spatial":
            if args["temporal_tag"] and args["temporal_tag"] != tag:
                stage = 1
            else:
                if args["storage_saving"]:
                    stage = 999
                else:
                    stage = args["temporal_stage"] + 1
        elif kind == "temporal":
            if args["spatial_tag"] and args["spatial_tag"] != tag:
                stage = 1
            else:
                if args["storage_saving"] or (args["spatial_stage"] == 999):
                    stage = 999
                else:
                    stage = args["spatial_stage"]
        else:
            stage = 0

        click.echo("-----------------------------------")
        click.echo(f"{tag} {stage:03} {kind}:")

        if self.force:
            return False

        if stage == 999:
            return False

        kind = dict(
            data="1data",
            find="2find",
            make="1spatial",
            temporal="2temporal",
            spatial="1spatial",
        )[kind]
        path = self.out_path(tag, stage, kind, "log.pickle")
        return os.path.exists(path)

    # DATA

    def data_tag(self, *args, **kwargs):
        return self.log(*args, **kwargs)["data_tag"]

    def nt(self, tag):
        return self.log(tag, 0, "1data")["nt"]

    def data(self, tag):
        path = self.out_path(tag, 0, "1data", "data.tfrecord")
        return load_tfrecord(path)

    def mask(self, tag):
        path = self.out_path(tag, 0, "1data", "mask.npy")
        return load_numpy(path)

    def avgt(self, tag):
        path = self.out_path(tag, 0, "1data", "avgt.npy")
        return load_numpy(f"{path}.npy")

    def avgx(self, tag):
        path = self.out_path(tag, 0, "1data", "avgx.npy")
        return load_numpy(f"{path}.npy")

    # Used Params

    def used_radius(self, tag):
        return self.log(tag, 0, "2find")["radius"]

    def used_distance(self, tag, stage, kind="1spatial"):
        return self.log(tag, stage, kind)["distance"]

    def used_dynamics(self, tag, stage, kind="2temporal"):
        return self.log(tag, stage, kind)["dynamics"]

    # Info

    def info(self, tag, stage, kind):
        path = self.out_path(tag, stage, kind, "info.csv")
        return load_csv(path)

    # Spatial

    def footprint(self, tag, stage, kind):
        path = self.out_path(tag, stage, kind, "footprint.npy")
        return load_numpy(path)

    def localx(self, tag, stage, kind):
        path = self.out_path(tag, stage, kind, "localx.npy")
        return load_numpy(path)

    # Temporal

    def spike(self, tag, stage, kind):
        path = self.out_path(tag, stage, kind, "spike.npy")
        return load_numpy(path)

    def localt(self, tag, stage, kind):
        path = self.out_path(tag, stage, kind, "localt.npy")
        return load_numpy(path)
