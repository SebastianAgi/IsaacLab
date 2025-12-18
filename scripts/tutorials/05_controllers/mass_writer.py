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
        # Mapping from prim path -> scalar (e.g. mass). Keys may not
        # exactly match the instance-id prim paths, so we also build a
        # normalized-name lookup for fuzzy matching.
        self.mass_lookup = mass_lookup or {}
        self._basic_writer = rep.BasicWriter(output_dir=output_dir, **basic_kwargs)
        
        mode = "friction" if use_friction else "mass"
        # self._scalar_dir = os.path.join(output_dir, mode)
        self._scalar_dir = output_dir
        # os.makedirs(self._scalar_dir, exist_ok=True)

        self._stage = get_current_stage()

        # Pre-compute a mapping from a normalized leaf-name of the prim
        # path (last component) to the scalar value. This allows us to
        # associate masses like "/World/envs/env_0/Object" with
        # instance-id labels such as
        # "/World/envs/env_0/Object/geometry/mesh" by matching the
        # unique "Object" token.
        self._name_to_scalar = {}
        for key, value in self.mass_lookup.items():
            leaf = os.path.basename(key.rstrip("/"))
            norm = self._normalize_token(leaf)
            if not norm:
                continue
            if norm in self._name_to_scalar and self._name_to_scalar[norm] != value:
                # If this happens, multiple different masses share the same
                # normalized name; we keep the first and warn once.
                print(
                    f"[MassWriter] Warning: multiple entries share normalized name '{norm}'. "
                    "Using the first one."
                )
                continue
            self._name_to_scalar[norm] = value

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
            # if scalar is not None and scalar != 0.0:
            #     print(f"[MassWriter] Prim: {prim_path}, ID: {inst_id}, Scalar: {scalar}, scalar type: {type(scalar)}")
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
        out_path = os.path.join(self._scalar_dir, f"mass_{frame_id:04d}.npy")
        np.save(out_path, scalar_map)

    # ------------------------------------------------------------------

    def _get_scalar(self, prim_path):
        # 1) Exact match on the full prim path.
        if prim_path in self.mass_lookup:
            return self.mass_lookup[prim_path]

        # 2) Walk up the prim hierarchy: many label paths point to a
        #    child (e.g. geometry/mesh). If any parent prim has a
        #    scalar, use that.
        parent = prim_path.rstrip("/")
        while "/" in parent:
            parent = parent.rsplit("/", 1)[0]
            if parent in self.mass_lookup:
                return self.mass_lookup[parent]

        # 3) Fallback: match by normalized leaf-name. We rely on the
        #    user-provided guarantee that the last component of the
        #    mass_lookup paths is unique across the scene. We then look
        #    for that component (normalized) in any segment of the
        #    instance-id prim path.
        segments = [seg for seg in prim_path.split("/") if seg]
        norm_segments = [self._normalize_token(seg) for seg in segments]
        for norm_seg in norm_segments:
            if norm_seg in self._name_to_scalar:
                return self._name_to_scalar[norm_seg]

        # If all strategies fail, fall back to zero (background).
        return 0.0

    @staticmethod
    def _normalize_token(token: str) -> str:
        """Normalize a prim-name token for fuzzy matching.

        Keeps only alphanumeric characters and lowercases them so that
        names like "link_0", "panda_link0" and "Link0" all normalize
        to comparable strings (e.g., "link0"). This makes it easier to
        associate physics prims with visual prims that share a core
        name but differ in decoration.
        """
        return "".join(ch for ch in token if ch.isalnum()).lower()

