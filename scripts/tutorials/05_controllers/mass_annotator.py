import omni.replicator.core as rep
from pxr import Usd, UsdPhysics
import omni.usd
import numpy as np

def compute_mass_value(instance_seg_img, use_friction):
    """Core logic for converting instance mask → mass/friction mask."""
    stage = omni.usd.get_context().get_stage()

    # Get mapping instance_id → prim path
    inst_annot = rep.annotators.get("instance_segmentation")
    instances = inst_annot.get_instances()   # dict: id → prim_path

    out = np.zeros_like(instance_seg_img, dtype=np.float32)

    for inst_id, prim_path in instances.items():
        prim = stage.GetPrimAtPath(prim_path)

        # mass or friction value
        if use_friction:
            mat_rel = prim.GetRelationship("physics:material:binding")
            if mat_rel and mat_rel.GetTargets():
                mat_prim = stage.GetPrimAtPath(mat_rel.GetTargets()[0])
                attr = mat_prim.GetAttribute("physics:staticFriction")
                val = float(attr.Get()) if attr and attr.HasValue() else 0.0
            else:
                val = 0.0
        else:
            mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
            if mass_api and mass_api.GetMassAttr().HasValue():
                val = float(mass_api.GetMassAttr().Get())
            else:
                val = 0.0

        out[instance_seg_img == inst_id] = val

    return out


# -------------------------------------------------------
# Register Scripted Annotator (Replicator 1.6+)
# -------------------------------------------------------
rep.annotators.register_script_annotator(
    name="mass_value",
    inputs=["instance_segmentation"],     # existing annotator we depend on
    output_dtype="float32",
    output_channels=1,
    compute_func=lambda data, use_friction=False: compute_mass_value(
        data["instance_segmentation"], use_friction
    ),
)
