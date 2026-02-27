"""Gaussian model: parameter initialization and PLY export."""

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree


def init_gaussians(
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    device: str = "cuda",
) -> dict[str, nn.Parameter]:
    """Initialize Gaussian parameters from a sparse point cloud.

    Args:
        points_xyz: (N, 3) point positions.
        points_rgb: (N, 3) point colors in [0, 1].
        device: Torch device.

    Returns:
        Dict of nn.Parameters: means, scales, quats, opacities, sh0, sh_rest.
    """
    n = len(points_xyz)

    # Positions
    means = torch.tensor(points_xyz, dtype=torch.float32, device=device)

    # Scales: initialize from local point density (log scale)
    # Use a simple heuristic: average distance to 3 nearest neighbors
    tree = KDTree(points_xyz)
    dists, _ = tree.query(points_xyz, k=4)  # k=4 because first is self
    avg_dist = dists[:, 1:].mean(axis=1)  # skip self
    avg_dist = np.clip(avg_dist, 1e-7, None)
    scales = torch.tensor(
        np.log(np.stack([avg_dist] * 3, axis=-1)),
        dtype=torch.float32,
        device=device,
    )

    # Rotations: identity quaternion (w, x, y, z)
    quats = torch.zeros(n, 4, dtype=torch.float32, device=device)
    quats[:, 0] = 1.0

    # Opacities: inverse sigmoid of 0.1
    opacity_init = 0.1
    raw_opacity = np.log(opacity_init / (1.0 - opacity_init))
    opacities = torch.full(
        (n,), raw_opacity, dtype=torch.float32, device=device
    )

    # Spherical harmonics: DC term from RGB, rest zeros
    # SH0 coefficient = (color - 0.5) / C0 where C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    sh0 = torch.tensor(
        (points_rgb - 0.5) / C0, dtype=torch.float32, device=device
    ).unsqueeze(1)  # (N, 1, 3)

    # Higher order SH coefficients (degree 3 -> 16 total, minus 1 DC = 15 rest)
    sh_rest = torch.zeros(n, 15, 3, dtype=torch.float32, device=device)

    params = {
        "means": nn.Parameter(means),
        "scales": nn.Parameter(scales),
        "quats": nn.Parameter(quats),
        "opacities": nn.Parameter(opacities),
        "sh0": nn.Parameter(sh0),
        "sh_rest": nn.Parameter(sh_rest),
    }

    print(f"Initialized {n} Gaussians")
    print(f"  Means: {means.shape}")
    print(f"  Scales: {scales.shape}")
    print(f"  Opacities: {opacities.shape}")
    print(f"  SH0: {sh0.shape}, SH rest: {sh_rest.shape}")

    return params


def save_checkpoint(
    params: dict[str, nn.Parameter],
    optimizers: dict[str, torch.optim.Optimizer],
    step: int,
    path: str,
) -> None:
    """Save training checkpoint (params + optimizer state + step)."""
    ckpt = {
        "step": step,
        "params": {k: v.data for k, v in params.items()},
        "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
    }
    torch.save(ckpt, path)
    print(f"Checkpoint saved: step {step} → {path}")


def load_checkpoint(
    path: str,
    device: str = "cuda",
) -> tuple[dict[str, nn.Parameter], dict, int]:
    """Load training checkpoint.

    Returns:
        params: Dict of nn.Parameters on device.
        optimizer_states: Dict of optimizer state_dicts.
        step: Training step to resume from.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    params = {}
    for k, v in ckpt["params"].items():
        params[k] = nn.Parameter(v.to(device))

    n = params["means"].shape[0]
    print(f"Loaded checkpoint: step {ckpt['step']}, {n} Gaussians from {path}")
    return params, ckpt["optimizers"], ckpt["step"]


def save_ply(params: dict[str, nn.Parameter], path: str) -> None:
    """Save Gaussian parameters to a PLY file.

    Args:
        params: Dict of Gaussian parameters (means, scales, quats, opacities, sh0, sh_rest).
        path: Output PLY file path.
    """
    means = params["means"].detach().cpu().numpy()
    scales = params["scales"].detach().cpu().numpy()
    quats = params["quats"].detach().cpu().numpy()
    opacities = params["opacities"].detach().cpu().numpy()
    sh0 = params["sh0"].detach().cpu().numpy().squeeze(1)  # (N, 3)
    sh_rest = params["sh_rest"].detach().cpu().numpy()  # (N, 15, 3)

    n = means.shape[0]
    normals = np.zeros_like(means)

    # Flatten SH rest: interleave (N, 15, 3) -> (N, 45)
    sh_rest_flat = sh_rest.reshape(n, -1)

    # Build structured array
    attrs = []
    attrs.extend([("x", "f4"), ("y", "f4"), ("z", "f4")])
    attrs.extend([("nx", "f4"), ("ny", "f4"), ("nz", "f4")])

    # SH DC (f_dc_0, f_dc_1, f_dc_2)
    for i in range(3):
        attrs.append((f"f_dc_{i}", "f4"))

    # SH rest
    for i in range(sh_rest_flat.shape[1]):
        attrs.append((f"f_rest_{i}", "f4"))

    attrs.append(("opacity", "f4"))

    for i in range(3):
        attrs.append((f"scale_{i}", "f4"))

    for i in range(4):
        attrs.append((f"rot_{i}", "f4"))

    dtype = np.dtype(attrs)
    elements = np.empty(n, dtype=dtype)

    elements["x"] = means[:, 0]
    elements["y"] = means[:, 1]
    elements["z"] = means[:, 2]
    elements["nx"] = normals[:, 0]
    elements["ny"] = normals[:, 1]
    elements["nz"] = normals[:, 2]

    for i in range(3):
        elements[f"f_dc_{i}"] = sh0[:, i]

    for i in range(sh_rest_flat.shape[1]):
        elements[f"f_rest_{i}"] = sh_rest_flat[:, i]

    elements["opacity"] = opacities

    for i in range(3):
        elements[f"scale_{i}"] = scales[:, i]

    for i in range(4):
        elements[f"rot_{i}"] = quats[:, i]

    ply_element = PlyElement.describe(elements, "vertex")
    PlyData([ply_element]).write(path)
    print(f"Saved {n} Gaussians to {path}")
