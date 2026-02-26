"""Export trained Gaussian splatting model to PLY format."""

import argparse
from pathlib import Path

import torch

from gaussian_model import save_ply


def load_checkpoint(ckpt_dir: str) -> dict:
    """Load Gaussian parameters from a checkpoint directory containing point_cloud.ply.

    Re-imports the PLY into the same param dict format used by training so it
    can be re-exported (e.g. after pruning or format conversion).
    """
    import numpy as np
    from plyfile import PlyData

    ply_path = Path(ckpt_dir) / "point_cloud.ply"
    if not ply_path.exists():
        ply_path = Path(ckpt_dir)
        if not ply_path.exists():
            raise FileNotFoundError(f"No PLY found at {ckpt_dir}")

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"]
    n = len(v["x"])

    means = torch.tensor(np.column_stack([v["x"], v["y"], v["z"]]), dtype=torch.float32)
    scales = torch.tensor(np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]]), dtype=torch.float32)
    quats = torch.tensor(np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]), dtype=torch.float32)
    opacities = torch.tensor(np.array(v["opacity"]), dtype=torch.float32)

    sh0 = torch.tensor(
        np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]),
        dtype=torch.float32,
    ).unsqueeze(1)

    rest_cols = sorted([name for name in v.data.dtype.names if name.startswith("f_rest_")],
                       key=lambda s: int(s.split("_")[-1]))
    if rest_cols:
        sh_rest = torch.tensor(
            np.column_stack([np.array(v[c]) for c in rest_cols]),
            dtype=torch.float32,
        ).reshape(n, -1, 3)
    else:
        sh_rest = torch.zeros(n, 15, 3, dtype=torch.float32)

    return {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(sh0),
        "sh_rest": torch.nn.Parameter(sh_rest),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export / re-export Gaussians to PLY")
    parser.add_argument("--input", required=True, help="Checkpoint dir or PLY path")
    parser.add_argument("--output", required=True, help="Output PLY path")
    args = parser.parse_args()

    params = load_checkpoint(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_ply(params, args.output)
