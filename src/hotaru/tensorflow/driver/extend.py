from os import (
    PathLike,
    fspath,
)

import numpy as np
import pandas as pd
import tensorflow as tf

from ..evaluate.utils import (
    calc_denseness,
    calc_overwrap,
)
from ..footprint.clean import clean_segment
from ..footprint.find import find_peak
from ..footprint.make import make_segment
from ..footprint.reduce import reduce_peak


class HotaruExtendMixin:
    """Extend"""

    def set_model_path(self, path: PathLike):
        self.path = path
        self._saved = None

    def save_stats(self):
        np.savez(self.path / "stats.npz", **self.stats._asdict())

    def load_stats(self):
        if not (self.path / "stats.npz").exists():
            return
        stats = np.load(self.path / "stats.npz")
        self._stats = stats["avgx"], stats["avgt"], stats["std"]

    def save(self, name, only_info=False):
        path = self.path / name
        path.mkdir(parents=True, exist_ok=True)
        if not only_info:
            module = tf.Module()
            module.footprint = tf.Variable(self.footprint.val_tensor())
            module.spike = tf.Variable(self.spike.val_tensor())
            module.localx = tf.Variable(self.localx.val_tensor())
            module.localt = tf.Variable(self.localt.val_tensor())
            tf.saved_model.save(module, fspath(path))
        self.info.to_csv(path / "info.csv")
        self._saved = name

    def load(self, name, only_info=False):
        if self._saved == name:
            return True
        path = self.path / name
        if not (path / "info.csv").exists():
            return False
        self.info = pd.read_csv(path / "info.csv", index_col=0)
        if not only_info:
            self.build(nk=self.info.shape[0])
            module = tf.saved_model.load(fspath(path))
            self.footprint.val = module.footprint
            self.spike.val = module.spike
            self.localx.val = module.localx
            self.localt.val = module.localt
        self._saved = name
        return True

    def set_radius_logspace(self, radius_min, radius_max, radius_num):
        self.radius = np.logspace(
            np.log10(radius_min),
            np.log10(radius_max),
            radius_num,
        )

    def update_extend_params(self, **kwargs):
        names = [
            "min_distance",
            "min_firmness",
            "thr_area",
            "max_nonzero",
            "max_overwrap",
            "max_denseness",
            "shard",
            "block_size",
            "batch",
        ]
        for k, v in kwargs.items():
            if k not in names:
                raise ValueError(f"unknown params: {k}")
            setattr(self, k, v)

    def find_peak(self, name=None, force=False):
        if force or not self.load(name, True):
            info = find_peak(
                self.imgs,
                self.mask,
                self.radius,
                self.shard,
                self.batch,
            )
            info = pd.DataFrame(info)
            info.sort_values("intensity", ascending=False, inplace=True)
            info.reset_index(drop=True, inplace=True)
            self.info = info
            self.save(name, only_info=True)

    def initialize(self, name=None, backup=False, force=False):
        if force or not self.load(name):
            self.reduce_peak()
            self.build()
            segment = make_segment(self.imgs, self.mask, self.info, self.batch)
            nk = np.count_nonzero(self.info.kind == "cell")
            self.footprint.val = segment[:nk]
            self.localx.val = segment[nk:]
            self.spike.clear(0)
            self.localt.clear(0)
            if backup:
                self.save(f"{name}_bk")
            self.info_backup = self.info
            self.post_initialize()
            self.save(name)

    def spatial_step(self, name=None, backup=False, force=False):
        if force or not self.load(name):
            self.fit_spatial()
            if backup:
                self.save(f"{name}_bk")
            self.info_backup = self.info
            self.post_spatial()
            self.save(name)

    def temporal_step(self, name=None, backup=False, force=False):
        if force or not self.load(name):
            self.fit_temporal()
            if backup:
                self.save(f"{name}_bk")
            self.info_backup = self.info
            self.post_temporal()
            self.save(name)

    def reduce_peak(self):
        min_radius = 1.01 * self.radius[0]
        max_radius = 0.99 * self.radius[-1]

        info = self.info

        if "kind" not in info:
            info.insert(0, "kind", "-")
            info.insert(1, "id", -1)

        cell = info.loc[info.radius < max_radius].copy()
        cell = reduce_peak(cell, self.min_distance, self.block_size).copy()
        cell = cell.loc[cell.radius > min_radius].copy()
        cell["kind"] = "cell"
        cell["id"] = np.arange(cell.shape[0])

        local = info.loc[info.radius >= max_radius].copy()
        local = reduce_peak(local, self.min_distance, self.block_size).copy()
        local["kind"] = "local"
        local["id"] = np.arange(local.shape[0])

        self.info = pd.concat([cell, local], axis=0)

    def post_initialize(self):
        info = self.info_backup

        cell = info[info.kind == "cell"].copy()
        if cell.shape[0] > 0:
            footprint = self.footprint.val_tensor()
            cell["nonzero"] = tf.math.count_nonzero(footprint > 0, axis=1).numpy()
            cell["overwrap"] = calc_overwrap(footprint > self.thr_area).numpy()

        local = info[info.kind == "local"].copy()
        if local.shape[0] > 0:
            localx = self.localx.val_tensor()
            local["nonzero"] = tf.math.count_nonzero(localx > 0, axis=1).numpy()
            local["overwrap"] = calc_overwrap(localx > self.thr_area).numpy()

        self.info = pd.concat([cell, local], axis=0)

    def post_spatial(self):
        info = self.info

        if "t" in info:
            info = info.drop(columns="t")
        if "intensity" in self.info:
            info = info.rename(columns=dict(intensity="firmness"))

        cell = info[info.kind == "cell"].copy()
        if cell.shape[0] > 0:
            footprint, scale, x, y, r, f = clean_segment(
                self.footprint.val_tensor(),
                self.mask,
                self.radius,
                self.batch,
            )
            self.footprint.val = footprint
            self.spike.val = self.spike.val_tensor() * scale[:, None]
            cell["x"] = x
            cell["y"] = y
            cell["radius"] = r
            cell["firmness"] = f
            cell["scale"] = tf.math.reduce_max(self.spike.val_tensor(), axis=1)

        local = info[info.kind == "local"].copy()
        if local.shape[0] > 0:
            localx, scale, x, y, r, f = clean_segment(
                self.localx.val_tensor(), self.mask, self.radius, self.batch
            )
            self.localx.val = localx
            self.localt.val = self.localt.val_tensor() * scale[:, None]
            local["x"] = x
            local["y"] = y
            local["radius"] = r
            local["firmness"] = f
            local["scale"] = tf.math.reduce_max(self.localt.val_tensor(), axis=1)

        self.info = pd.concat([cell, local], axis=0)
        self.sort()

    def post_temporal(self):
        info = self.info

        spike = self.spike.val_tensor()
        cell = info[info.kind == "cell"].copy()
        cell["scale"] = tf.math.reduce_max(spike, axis=1).numpy()
        cell["denseness"] = calc_denseness(spike).numpy()

        localt = self.localt.val_tensor()
        local = info[info.kind == "local"].copy()
        local["scale"] = tf.math.reduce_max(localt, axis=1).numpy()
        local["denseness"] = calc_denseness(localt).numpy()

        self.info = pd.concat([cell, local], axis=0)
        self.sort()

    def sort(self):
        min_radius = 1.01 * self.radius[0]
        max_radius = 0.99 * self.radius[-1]
        max_overwrap = self.max_overwrap
        max_denseness = self.max_denseness
        min_firmness = self.min_firmness
        max_nonzero = self.max_nonzero
        info = self.info

        if "old_kind" not in info:
            info.insert(2, "old_kind", "-")
            info.insert(3, "old_id", -1)
        info["old_kind"] = info["kind"]
        info["old_id"] = info["id"]

        select_cond = (
            (info.radius > min_radius)
            & (info.nonzero > 1)
            & (info.overwrap < max_overwrap)
        )

        self.removed = info.loc[~select_cond].copy()
        self.removed["kind"] = "removed"
        info = info.loc[select_cond].copy()

        cell_cond = (info.radius < max_radius) & (info.nonzero < max_nonzero)
        if "denseness" in info:
            cell_cond &= info.denseness < max_denseness
        if "firmness" in info:
            cell_cond &= info.firmness > min_firmness

        cell = info.loc[cell_cond].copy()
        if "firmness" in info:
            cell.sort_values("firmness", ascending=False, inplace=True)
        cell["kind"] = "cell"
        cell["id"] = np.arange(cell.shape[0])

        local = info.loc[~cell_cond].copy()
        if "firmness" in info:
            local.sort_values("firmness", ascending=False, inplace=True)
        local["kind"] = "local"
        local["id"] = np.arange(local.shape[0])

        self.local_to_cell = cell.loc[info.old_kind == "local"].copy()
        self.cell_to_local = local.loc[info.old_kind == "cell"].copy()
        info = pd.concat([cell, local], axis=0)

        footprint = self.footprint.val_tensor()
        spike = self.spike.val_tensor()
        spike_ = self._spike_to_calcium(spike)
        localx = self.localx.val_tensor()
        localt = self.localt.val_tensor()
        localt_ = self._calcium_to_spike(localt)

        cell_id = cell[["old_kind", "old_id"]].to_numpy()
        if cell_id.shape[0] == 0:
            self.footprint.clear(0)
            self.spike.clear(0)
        else:
            self.footprint.val = [
                footprint[i] if k == "cell" else localx[i] for k, i in cell_id
            ]
            self.spike.val = [
                spike[i] if k == "cell" else localt_[i] for k, i in cell_id
            ]

        local_id = local[["old_kind", "old_id"]].to_numpy()
        if local_id.shape[0] == 0:
            self.localx.clear(0)
            self.localt.clear(0)
        else:
            self.localx.val = [
                footprint[i] if k == "cell" else localx[i] for k, i in local_id
            ]
            self.localt.val = [
                spike_[i] if k == "cell" else localt[i] for k, i in local_id
            ]

        self.info = info
