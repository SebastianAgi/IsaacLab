import json
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image


def sanity_check_mass_mask(
    rgb_path: str,
    seg_path: str,
    mapping_path: str,
    mass_path: str,
    verbose: bool = True,
):
    """
    Sanity-check that:
      - Instance segmentation colors correspond to entries in mapping.
      - Mass mask values are consistent per instance ID.
      - Visual overlays look correct.

    Args:
        rgb_path: path to RGB image (.png)
        seg_path: path to instance_id_segmentation (.png)
        mapping_path: mapping JSON saved by your writer
        mass_path: path to .npy scalar mask
    """

    rgb = np.array(Image.open(rgb_path))
    seg = np.array(Image.open(seg_path))
    mapping = json.load(open(mapping_path, "r"))
    mass = np.load(mass_path)

    print("---- Loaded ----")
    print("RGB:", rgb.shape)
    print("Seg:", seg.shape, seg.dtype)
    print("Mass:", mass.shape, mass.dtype)
    print("Mapping entries:", len(mapping))

    # ----------------------------------------------------------
    # 1. Convert segmentation RGBA → integer IDs
    # ----------------------------------------------------------
    if seg.shape[-1] == 4:
        seg_rgba = seg.astype(np.int64)
        seg_ids = (
            seg_rgba[..., 0]
            + (seg_rgba[..., 1] << 8)
            + (seg_rgba[..., 2] << 16)
            + (seg_rgba[..., 3] << 24)
        )
    else:
        raise ValueError("Unexpected segmentation shape, expected RGBA.")

    # ----------------------------------------------------------
    # 2. Compute per-instance mean mass (to verify consistency)
    # ----------------------------------------------------------
    unique_ids = np.unique(seg_ids)
    print("\nUnique instance IDs in segmentation:", unique_ids)

    # Compute mean mass per ID
    id_to_mean_mass = {}
    for inst_id in unique_ids:
        mask = seg_ids == inst_id
        values = mass[mask]
        id_to_mean_mass[inst_id] = float(np.mean(values)) if values.size > 0 else None

    print("\nInstance ID → mean mass:")
    for k, v in id_to_mean_mass.items():
        print(f"  {k}: {v}")

    # ----------------------------------------------------------
    # 3. Check if all instance IDs exist in mapping
    # ----------------------------------------------------------
    print("\nChecking ID presence in mapping.json:")
    for inst_id in unique_ids:
        key = str(inst_id)
        if key not in mapping:
            print(f"  WARNING: ID {inst_id} missing in mapping file!")
        else:
            print(f"  OK: {inst_id} → {mapping[key]}")

    # ----------------------------------------------------------
    # 4. Visual overlays
    # ----------------------------------------------------------
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 3, 1)
    plt.title("RGB")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Instance Seg IDs")
    plt.imshow(seg_ids, cmap="tab20")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Mass (float mask)")
    plt.imshow(mass, cmap="viridis")
    plt.colorbar()
    plt.axis("off")

    plt.show()

    print("\nSanity check complete.")

if __name__ == "__main__":
    # Example usage
    data_dir = Path("/home/Sebastian/IsaacLab/scripts/tutorials/05_controllers/output/camera/")
    sanity_check_mass_mask(
        rgb_path=str(data_dir / "rgb_100_0000.png"),
        seg_path=str(data_dir / "instance_id_segmentation_100_0000.png"),
        mapping_path=str(data_dir / "instance_id_segmentation_mapping_100_0000.json"),
        mass_path=str(data_dir / "mass" / "000100.npy"),
    )