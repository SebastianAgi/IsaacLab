#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def save_mass_png(
    npy_path: str,
    out_path: str = None,
    cmap: str = "viridis",
    normalize: bool = True,
    scale_factor: float = 255.0,
):
    """
    Convert a mass map stored as .npy (HxW float32) into a visible PNG.

    Args:
        npy_path: input file (.npy)
        out_path: output PNG path; if None, creates next to input
        cmap: matplotlib colormap name (e.g., viridis, jet, plasma)
        normalize: if True, mass is scaled to [0, scale_factor]
        scale_factor: upper bound of normalized range (default 255)

    Returns:
        Path to the generated PNG.
    """

    npy_path = Path(npy_path)
    mass = np.load(npy_path)

    if mass.ndim != 2:
        raise ValueError(f"Expected 2D mass image but got shape {mass.shape}")

    # Compute output filename
    if out_path is None:
        out_path = npy_path.with_suffix(".png")
    out_path = Path(out_path)

    # Normalize for visualization
    if normalize:
        m_min = np.min(mass)
        m_max = np.max(mass)
        if m_max > m_min:
            norm = (mass - m_min) / (m_max - m_min)
        else:
            norm = np.zeros_like(mass)
    else:
        norm = mass / scale_factor

    # Apply colormap
    colormap = cm.get_cmap(cmap)
    colored = colormap(norm)  # (H, W, 4) float 0–1

    # Convert to 8-bit PNG
    colored_uint8 = (colored[..., :3] * 255).astype(np.uint8)
    img = Image.fromarray(colored_uint8)

    img.save(out_path)
    print(f"[Mass PNG] Saved: {out_path}")
    return out_path


def save_mass_exr(npy_path: str, out_path: str = None):
    """
    Save the mass map as a true-high-precision EXR (float32).
    """
    try:
        import imageio.v3 as iio
    except ImportError:
        raise ImportError("Install imageio[ffmpeg]: pip install imageio[ffmpeg]")

    npy_path = Path(npy_path)
    mass = np.load(npy_path).astype(np.float32)

    if out_path is None:
        out_path = npy_path.with_suffix(".exr")
    out_path = Path(out_path)

    iio.imwrite(out_path, mass)
    print(f"[Mass EXR] Saved: {out_path}")
    return out_path


def batch_convert(folder: str, cmap="viridis"):
    """
    Convert all .npy files in a folder to PNG visualizations.
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.npy"))
    print(f"Found {len(files)} .npy files.")

    for f in files:
        save_mass_png(f, cmap=cmap)


def main():
    parser = argparse.ArgumentParser(description="Convert mass .npy to PNG/EXR")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to .npy mass file")
    parser.add_argument("--cmap", "-c", type=str, default="viridis", help="Matplotlib colormap")
    parser.add_argument("--exr", action="store_true", help="Also output EXR")
    args = parser.parse_args()

    png_path = save_mass_png(args.input, cmap=args.cmap)
    if args.exr:
        save_mass_exr(args.input)

    print("Done.")


if __name__ == "__main__":
    main()


# Example usage:
# python save_mass_image.py --input /path/to/mass/000038.npy --cmap plasma