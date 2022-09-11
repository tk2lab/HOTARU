import os

import click
import numpy as np
import tensorflow as tf

from ..io.csv import load_csv
from ..io.csv import save_csv
from ..io.json import load_json
from ..io.json import save_json
from ..io.numpy import load_numpy
from ..io.numpy import save_numpy
from ..io.tfrecord import load_tfrecord
from ..io.tfrecord import save_tfrecord
from ..io.tiff import save_tiff
from ..train.model import HotaruModel as Model
from ..util.distribute import MirroredStrategy
from .progress import ProgressCallback


class Obj:
    """HOTARU CLI Object."""

    def __init__(self):
        self._log = {}
        self._model_tag = None

    def __enter__(self):
        self.strategy = MirroredStrategy()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.strategy.close()

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
            prev_log = self.log("1data", data_tag, 0)
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

    def get_radius(self, radius_type, radius_min, radius_max, radius_num):
        if radius_type == "linear":
            radius_func = np.linspace
        elif radius_type == "log":
            radius_func = np.logspace
            radius_min = np.log10(radius_min)
            radius_max = np.log10(radius_max)
        return radius_func(radius_min, radius_max, radius_num)

    def callbacks(self, label, log_dir):
        return [
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                update_freq="batch",
                write_graph=False,
            ),
            ProgressCallback(label),
        ]

    # SAVE

    def out_path(self, kind=None, tag=None, stage=None):
        if stage is None:
            stage = ""
        elif isinstance(stage, int):
            stage = f"_{stage:03}"
        os.makedirs(f"{self.workdir}/{kind}", exist_ok=True)
        return f"{self.workdir}/{kind}/{tag}{stage}"

    def save_tfrecord(self, data, kind=None, tag=None, stage=None):
        out_path = self.out_path(kind, tag, stage)
        save_tfrecord(f"{out_path}.tfrecord", data)

    def save_numpy(self, data, kind=None, tag=None, stage=None):
        out_path = self.out_path(kind, tag, stage)
        save_numpy(f"{out_path}.npy", data)

    def save_csv(self, data, kind=None, tag=None, stage=None):
        out_path = self.out_path(kind, tag, stage)
        save_csv(f"{out_path}.csv", data)

    def save_tiff(self, data, kind=None, tag=None, stage=None):
        out_path = self.out_path(kind, tag, stage)
        save_tiff(f"{out_path}.tif", data)

    # SUMMARY

    def summary_path(self, kind=None, tag=None, stage=None):
        if stage is None:
            stage = ""
        elif isinstance(stage, int):
            stage = f"_{stage:03}"
        return f"{self.workdir}/summary/{tag}/{kind}{stage}"

    # LOG

    def save_log(self, log, kind, tag, stage):
        os.makedirs(f"{self.workdir}/log", exist_ok=True)
        path = f"{self.workdir}/log/{tag}_{stage:03}_{kind}.json"
        save_json(path, log)
        self._log[path] = log

    def log(self, kind, tag, stage):
        path = f"{self.workdir}/log/{tag}_{stage:03}_{kind}.json"
        if path not in self._log:
            self._log[path] = load_json(path)
        return self._log[path]

    def can_skip(self, kind, tag, **args):
        if kind == "temporal":
            if args["segment_tag"] != tag:
                stage = 1
            else:
                if args["storage_saving"] or (args["segment_stage"] == 999):
                    stage = 999
                else:
                    stage = args["segment_stage"] + 1
        elif kind == "spatial":
            if args["spike_tag"] != tag:
                stage = 1
            else:
                if args["storage_saving"]:
                    stage = 999
                else:
                    stage = args["spike_stage"]
        elif kind == "clean":
            if args["footprint_tag"] != tag:
                stage = 1
            else:
                if args["storage_saving"]:
                    stage = 999
                else:
                    stage = args["footprint_stage"]
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
            make="3segment",
            temporal="1temporal",
            spatial="2spatial",
            clean="3segment",
        )[kind]
        path = f"{self.workdir}/log/{tag}_{stage:03}_{kind}.json"
        return os.path.exists(path)

    def need_exec(self):
        kind = self.kind
        if self.force:
            return True
        if kind == "output":
            return True
        if kind in ("temporal", "spatial", "clean"):
            if not isinstance(self.stage, int):
                return True
        path = self.log_path()
        return not os.path.exists(path)

    # DATA

    def data(self, tag):
        path = self.out_path("data", tag, "_data")
        return load_tfrecord(f"{path}.tfrecord")

    def mask(self, tag):
        path = self.out_path("data", tag, "_mask")
        return load_numpy(f"{path}.npy")

    def avgt(self, tag):
        path = self.out_path("data", tag, "argt")
        return load_numpy(f"{path}.npy")

    def avgx(self, tag):
        path = self.out_path("data", tag, "argx")
        return load_numpy(f"{path}.npy")

    # FIND

    def peaks(self, tag, stage="-find"):
        path = self.out_path("peak", tag, stage)
        return load_csv(f"{path}.csv")

    def used_radius_args(self, tag, stage=0):
        if stage == 0:
            log = self.log("2find", tag, stage)
        else:
            log = self.log("3segment", tag, stage)
        return {k: v for k, v in log.items() if k[:6] == "radius"}

    # MAKE

    def segment(self, tag, stage):
        path = self.out_path("segment", tag, stage)
        if stage == "_curr":
            if not os.path.exists(f"{path}.npy"):
                path = self.out_path("segment", tag, "_000")
        return load_numpy(f"{path}.npy")

    def localx(self, tag, stage):
        path = self.out_path("localx", tag, stage)
        if stage == "_curr":
            if not os.path.exists(f"{path}.npy"):
                path = self.out_path("localx", tag, "_000")
        return load_numpy(f"{path}.npy")

    def index(self, tag, stage):
        path = self.out_path("peak", tag, stage)
        if stage == "_curr":
            if not os.path.exists(f"{path}.csv"):
                path = self.out_path("peak", tag, "_000")
        return load_csv(f"{path}.csv").query('accept == "yes"').index

    def used_distance(self, tag, stage=0):
        return self.log("3segment", tag, stage=stage)["distance"]

    # Temporal

    def spike(self, tag, stage):
        path = self.out_path("spike", tag, stage)
        return load_numpy(f"{path}.npy")

    def localt(self, tag, stage):
        path = self.out_path("localt", tag, stage)
        if stage == "_curr":
            if not os.path.exists(f"{path}.npy"):
                path = self.out_path("localt", tag, "_000")
        return load_numpy(f"{path}.npy")

    def used_tau(self, tag, stage):
        log = self.log("1temporal", tag, stage)
        return dict(
            tau1=log["tau1"],
            tau2=log["tau2"],
        )

    # Spatial

    def footprint(self, tag, stage):
        path = self.out_path("footprint", tag, stage)
        return load_numpy(f"{path}.npy")

    def localx0(self, tag, stage):
        path = self.out_path("localx0", tag, stage)
        if stage == "_curr":
            if not os.path.exists(f"{path}.npy"):
                path = self.out_path("localx0", tag, "_000")
        return load_numpy(f"{path}.npy")
