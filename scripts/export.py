"""Export trained Gaussian splatting model to PLY format."""

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def export_ply(gaussians: dict, output_path: str) -> None:
    """Export Gaussian parameters to a .ply file.

    Args:
        gaussians: Dict with keys 'xyz', 'opacity', 'scales', 'rotations', 'sh'.
        output_path: Path to write the .ply file.
    """
    xyz = gaussians["xyz"]  # (N, 3)
    normals = np.zeros_like(xyz)  # placeholder normals
    opacities = gaussians["opacity"]  # (N, 1)
    scales = gaussians["scales"]  # (N, 3)
    rotations = gaussians["rotations"]  # (N, 4)
    sh = gaussians["sh"]  # (N, C) spherical harmonics coefficients

    n = xyz.shape[0]

    attrs = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("opacity", "f4"),
    ]

    for i in range(scales.shape[1]):
        attrs.append((f"scale_{i}", "f4"))

    for i in range(rotations.shape[1]):
        attrs.append((f"rot_{i}", "f4"))

    for i in range(sh.shape[1]):
        attrs.append((f"f_rest_{i}", "f4"))

    dtype = np.dtype(attrs)
    elements = np.empty(n, dtype=dtype)

    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["nx"] = normals[:, 0]
    elements["ny"] = normals[:, 1]
    elements["nz"] = normals[:, 2]
    elements["opacity"] = opacities[:, 0]

    for i in range(scales.shape[1]):
        elements[f"scale_{i}"] = scales[:, i]

    for i in range(rotations.shape[1]):
        elements[f"rot_{i}"] = rotations[:, i]

    for i in range(sh.shape[1]):
        elements[f"f_rest_{i}"] = sh[:, i]

    ply_element = PlyElement.describe(elements, "vertex")
    PlyData([ply_element]).write(output_path)
    print(f"Exported {n} Gaussians to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Gaussians to PLY")
    parser.add_argument("--input", required=True, help="Path to trained model dir")
    parser.add_argument("--output", required=True, help="Output PLY path")
    args = parser.parse_args()

    print("TODO: Load trained model and call export_ply()")
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
