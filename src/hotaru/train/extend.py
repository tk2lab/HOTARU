import numpy as np
import pandas as pd
import tensorflow as tf

from ..evaluate.utils import calc_denseness
from ..evaluate.utils import calc_overwrap
from ..footprint.clean import clean_segment
from ..footprint.find import find_peak
from ..footprint.make import make_segment
from ..footprint.reduce import reduce_peak
from .model import HotaruModel


class ExtendHotaruModel(HotaruModel):
    """Variable"""

    def find_peak(self, radius, shard, batch):
        self.info = find_peak(self.imgs, self.mask, radius, shard, batch)

    def reduce_peak(self, min_radius, max_radius, min_distance, block_size):
        info = self.info

        info.insert(0, "kind", "-")
        info.insert(1, "id", -1)

        cell = info.loc[info.radius < max_radius].copy()
        cell = reduce_peak(cell, min_distance, block_size).copy()
        cell = cell.loc[cell.radius > min_radius].copy()
        cell["kind"] = "cell"
        cell["id"] = np.arange(cell.shape[0])

        local = info.loc[info.radius >= max_radius].copy()
        local = reduce_peak(local, min_distance, block_size).copy()
        local["kind"] = "local"
        local["id"] = np.arange(local.shape[0])

        self.info = pd.concat([cell, local], axis=0)

    def initialize(self, batch):
        self.build_and_compile()
        segment = make_segment(self.imgs, self.mask, self.info, batch)
        nk = np.count_nonzero(self.info.kind == "cell")
        self.footprint.val = segment[:nk]
        self.localx.val = segment[nk:]
        self.spike.clear(0)
        self.localt.clear(0)

    def spatial_step(self, batch, **kwargs):
        self.build_and_compile()
        spike = self.spike.val_tensor()
        self.spike.val = spike / tf.math.reduce_max(
            spike, axis=1, keepdims=True
        )
        localt = self.localt.val_tensor()
        self.localt.val = localt / tf.math.reduce_max(
            localt, axis=1, keepdims=True
        )
        model = self.spatial_model
        model.prepare(batch=batch)
        model.fit(**kwargs)

    def temporal_step(self, batch, **kwargs):
        self.build_and_compile()
        model = self.temporal_model
        model.prepare(batch=batch)
        model.fit(**kwargs)

    def post_initialize(self, thr_area):
        info = self.info

        cell = info[info.kind == "cell"].copy()
        if cell.shape[0] > 0:
            footprint = self.footprint.val_tensor()
            cell["nonzero"] = tf.math.count_nonzero(
                footprint > 0, axis=1
            ).numpy()
            cell["overwrap"] = calc_overwrap(footprint > thr_area).numpy()

        local = info[info.kind == "local"].copy()
        if local.shape[0] > 0:
            localx = self.localx.val_tensor()
            local["nonzero"] = tf.math.count_nonzero(
                localx > 0, axis=1
            ).numpy()
            local["overwrap"] = calc_overwrap(localx > thr_area).numpy()

        self.info = pd.concat([cell, local], axis=0)

    def post_spatial(self, radius, thr_area, batch):
        info = self.info

        if "t" in info:
            info = info.drop(columns="t")
        if "intensity" in self.info:
            info = info.rename(columns=dict(intensity="firmness"))

        cell = info[info.kind == "cell"].copy()
        if cell.shape[0] > 0:
            footprint, scale, x, y, r, f = clean_segment(
                self.footprint.val_tensor(), self.mask, radius, batch
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
                self.localx.val_tensor(), self.mask, radius, batch
            )
            self.localx.val = localx
            self.localt.val = self.localt.val_tensor() * scale[:, None]
            local["x"] = x
            local["y"] = y
            local["radius"] = r
            local["firmness"] = f
            local["scale"] = tf.math.reduce_max(
                self.localt.val_tensor(), axis=1
            )

        self.info = pd.concat([cell, local], axis=0)
        self.post_initialize(thr_area)

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

    def sort(
        self,
        min_radius,
        max_radius,
        min_firmness,
        max_nonzero,
        max_overwrap,
        max_denseness,
    ):
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
        spike_ = self.spike_to_calcium(spike)
        localx = self.localx.val_tensor()
        localt = self.localt.val_tensor()
        localt_ = self.calcium_to_spike(localt)

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

    def summary(self):
        for kind in ["cell", "local"]:
            print(self.info[self.info.kind == kind])
