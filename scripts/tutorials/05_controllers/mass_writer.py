# mass_writer.py

import os
from typing import Dict, Any
import numpy as np

from pxr import UsdPhysics, PhysxSchema
from isaaclab.sim.utils.stage import get_current_stage
import omni.replicator.core as rep


class MassWriter(rep.Writer):

    def __init__(self, 
                 output_dir, 
                 use_friction=False, 
                 width=640, 
                 height=480, 
                 mass_lookup=None,
                 **basic_kwargs):
        super().__init__()

        self.width = width
        self.height = height
        self.use_friction = use_friction
        self.mass_lookup = mass_lookup or {}
        self._basic_writer = rep.BasicWriter(output_dir=output_dir, **basic_kwargs)

        mode = "friction" if use_friction else "mass"
        self._scalar_dir = os.path.join(output_dir, mode)
        os.makedirs(self._scalar_dir, exist_ok=True)

        self._stage = get_current_stage()

    # ------------------------------------------------------------------

    def write(self, data: Dict[str, Any]):
        self._basic_writer.write(data)

        try:
            self._write_scalar_map(data)
        except Exception as e:
            print(f"[MassWriter] Failed to write mass map: {e}")

    # ------------------------------------------------------------------

    def _write_scalar_map(self, data):
        annotators = data.get("annotators", {})
        seg_entry = annotators.get("instance_id_segmentation_fast")
        if seg_entry is None:
            return

        rp = seg_entry["render_product"]
        seg_img = np.asarray(rp["data"])
        id_to_labels = rp["idToLabels"]

        # seg_img must be (H, W, 4)
        if seg_img.shape != (self.height, self.width, 4):
            raise ValueError(f"Segmentation shape mismatch: {seg_img.shape}")

        # -------------------------------------------
        # Convert RGBA → uint32 instance IDs
        # -------------------------------------------
        r = seg_img[...,0].astype(np.uint32)
        g = seg_img[...,1].astype(np.uint32)
        b = seg_img[...,2].astype(np.uint32)
        a = seg_img[...,3].astype(np.uint32)

        id_img = r | (g << 8) | (b << 16) | (a << 24)

        # -------------------------------------------
        # Convert idToLabels keys into uint32 IDs
        # -------------------------------------------
        id_to_scalar = {}

        for key, prim_path in id_to_labels.items():

            # Key may be a tuple or string
            if isinstance(key, tuple):
                r,g,b,a = key
            else:
                r,g,b,a = (int(x) for x in key.strip("() ").split(","))

            inst_id = (
                (r & 0xFF)
                | ((g & 0xFF) << 8)
                | ((b & 0xFF) << 16)
                | ((a & 0xFF) << 24)
            )

            scalar = self._get_scalar(prim_path)
            if scalar is not None and scalar != 0.0:
                print(f"[MassWriter] Prim: {prim_path}, ID: {inst_id}, Scalar: {scalar}, scalar type: {type(scalar)}")
            id_to_scalar[inst_id] = scalar if scalar is not None else 0.0

        # -------------------------------------------
        # Rasterize scalar map
        # -------------------------------------------
        scalar_map = np.zeros((self.height, self.width), dtype=np.float32)

        for inst_id, scalar in id_to_scalar.items():
            mask = (id_img == inst_id)
            scalar_map[mask] = scalar

        # -------------------------------------------
        # Save
        # -------------------------------------------
        frame_id = int(data["trigger_outputs"]["on_time"])
        out_path = os.path.join(self._scalar_dir, f"{frame_id:06d}.npy")
        np.save(out_path, scalar_map)

    # ------------------------------------------------------------------

    def _get_scalar(self, prim_path):
        return self.mass_lookup.get(prim_path, 0.0)

