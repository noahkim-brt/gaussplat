"""Train 3D Gaussian Splatting model using gsplat."""

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from gsplat import DefaultStrategy, rasterization
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_loader import cameras_to_tensors, compute_scene_scale, load_colmap_scene
from gaussian_model import init_gaussians, save_ply

# ─── SSIM ────────────────────────────────────────────────────────────────────


def _gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
    return gauss / gauss.sum()


def _ssim(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    """Compute SSIM between two images. Both (H, W, 3) float tensors."""
    # Rearrange to (1, 3, H, W)
    img1 = img1.permute(2, 0, 1).unsqueeze(0)
    img2 = img2.permute(2, 0, 1).unsqueeze(0)
    C = img1.shape[1]

    kernel_1d = _gaussian_kernel_1d(window_size, 1.5).to(img1.device)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    window = kernel_2d.expand(C, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


# ─── Learning rate schedule ──────────────────────────────────────────────────


def lr_exponential_decay(
    step: int,
    lr_init: float,
    lr_final: float,
    max_steps: int,
) -> float:
    """Exponential learning rate decay."""
    t = min(step / max_steps, 1.0)
    log_lerp = math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)
    return log_lerp


# ─── Training ────────────────────────────────────────────────────────────────


def train(
    scene_dir: str,
    output_dir: str,
    model_idx: int = 0,
    iterations: int = 30_000,
    resolution_scale: int = 1,
    sh_degree: int = 3,
    lr_means: float = 1.6e-4,
    lr_scales: float = 5e-3,
    lr_quats: float = 1e-3,
    lr_opacities: float = 5e-2,
    lr_sh0: float = 2.5e-3,
    lr_sh_rest: float = 1.25e-4,
    lr_means_final: float = 1.6e-6,
    lambda_dssim: float = 0.2,
    save_iterations: list[int] | None = None,
    log_interval: int = 100,
) -> None:
    """Train a 3D Gaussian Splatting model.

    Args:
        scene_dir: Scene directory with sparse/ and images/ subdirs.
        output_dir: Directory to save trained model.
        model_idx: Which COLMAP model to use.
        iterations: Number of training iterations.
        resolution_scale: Downscale factor for images.
        sh_degree: Max spherical harmonics degree.
        lr_means: Learning rate for positions.
        lr_scales: Learning rate for scales.
        lr_quats: Learning rate for rotations.
        lr_opacities: Learning rate for opacities.
        lr_sh0: Learning rate for SH DC term.
        lr_sh_rest: Learning rate for SH higher-order terms.
        lr_means_final: Final learning rate for positions (exponential decay).
        lambda_dssim: Weight for SSIM loss component.
        save_iterations: Iterations at which to save checkpoints.
        log_interval: Logging interval.
    """
    if save_iterations is None:
        save_iterations = [7_000, 30_000]

    device = "cuda"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading scene data...")
    cameras, points_xyz, points_rgb = load_colmap_scene(
        scene_dir, model_idx=model_idx, resolution_scale=resolution_scale
    )
    viewmats, Ks, gt_images, widths, heights = cameras_to_tensors(cameras, device)
    scene_scale = compute_scene_scale(points_xyz)
    num_views = len(cameras)
    print(f"Scene scale: {scene_scale:.4f}")
    print(f"Image resolution: {widths[0]}x{heights[0]}")

    # ── Initialize Gaussians ──────────────────────────────────────────────
    params = init_gaussians(points_xyz, points_rgb, device)

    # ── Optimizers (one per parameter, required by gsplat strategy) ───────
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=lr_means, eps=1e-15),
        "scales": torch.optim.Adam([params["scales"]], lr=lr_scales, eps=1e-15),
        "quats": torch.optim.Adam([params["quats"]], lr=lr_quats, eps=1e-15),
        "opacities": torch.optim.Adam(
            [params["opacities"]], lr=lr_opacities, eps=1e-15
        ),
        "sh0": torch.optim.Adam([params["sh0"]], lr=lr_sh0, eps=1e-15),
        "sh_rest": torch.optim.Adam([params["sh_rest"]], lr=lr_sh_rest, eps=1e-15),
    }

    # ── Densification strategy ────────────────────────────────────────────
    strategy = DefaultStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    # ── Tensorboard ───────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(output_path / "tb"))

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\nStarting training for {iterations} iterations...")
    print(f"  Views: {num_views}")
    print(f"  Initial Gaussians: {params['means'].shape[0]}")
    print(f"  Loss: (1-{lambda_dssim})*L1 + {lambda_dssim}*SSIM")

    pbar = tqdm(range(1, iterations + 1), desc="Training")
    running_loss = 0.0
    t0 = time.time()

    for step in pbar:
        # Random view selection
        idx = random.randint(0, num_views - 1)
        viewmat = viewmats[idx : idx + 1]  # (1, 4, 4)
        K = Ks[idx : idx + 1]  # (1, 3, 3)
        gt_image = gt_images[idx]  # (H, W, 3)
        W, H = widths[idx], heights[idx]

        # Update position learning rate (exponential decay)
        lr_means_current = lr_exponential_decay(
            step, lr_means, lr_means_final, iterations
        )
        for pg in optimizers["means"].param_groups:
            pg["lr"] = lr_means_current

        # Activate SH bands progressively
        sh_degree_current = min(
            sh_degree, step // (iterations // (sh_degree + 1))
        )

        # Assemble SH coefficients
        colors = torch.cat([params["sh0"], params["sh_rest"]], dim=1)  # (N, 16, 3)

        # Render
        rendered, rendered_alpha, info = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            sh_degree=sh_degree_current,
            packed=True,
            absgrad=False,
            render_mode="RGB",
        )

        rendered_image = rendered[0]  # (H, W, 3)

        # Loss: L1 + SSIM
        l1_loss = F.l1_loss(rendered_image, gt_image)
        ssim_loss = 1.0 - _ssim(rendered_image, gt_image)
        loss = (1.0 - lambda_dssim) * l1_loss + lambda_dssim * ssim_loss

        # Densification callbacks
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        # Backward
        loss.backward()

        # Densification callbacks (post-backward: split, clone, prune)
        strategy.step_post_backward(params, optimizers, strategy_state, step, info, packed=True)

        # Optimizer step
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        # Logging
        running_loss += loss.item()
        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            n_gaussians = params["means"].shape[0]
            elapsed = time.time() - t0
            pbar.set_postfix(
                loss=f"{avg_loss:.5f}",
                n_gs=f"{n_gaussians // 1000}k",
                lr=f"{lr_means_current:.2e}",
            )
            writer.add_scalar("train/loss", avg_loss, step)
            writer.add_scalar("train/l1_loss", l1_loss.item(), step)
            writer.add_scalar("train/ssim_loss", ssim_loss.item(), step)
            writer.add_scalar("train/num_gaussians", n_gaussians, step)
            writer.add_scalar("train/lr_means", lr_means_current, step)
            running_loss = 0.0

        # Save checkpoints
        if step in save_iterations:
            ckpt_dir = output_path / "point_cloud" / f"iteration_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ply_path = str(ckpt_dir / "point_cloud.ply")
            save_ply(params, ply_path)

    # ── Final save ────────────────────────────────────────────────────────
    total_time = time.time() - t0
    final_dir = output_path / "point_cloud" / f"iteration_{iterations}"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_ply(params, str(final_dir / "point_cloud.ply"))

    writer.close()

    print(f"\nTraining complete in {total_time / 60:.1f} minutes")
    print(f"  Final Gaussians: {params['means'].shape[0]}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting")
    parser.add_argument("--scene", required=True, help="Scene directory")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--model-idx", type=int, default=0, help="COLMAP model index")
    parser.add_argument(
        "--iterations", type=int, default=30_000, help="Training iterations"
    )
    parser.add_argument(
        "--resolution-scale", type=int, default=1, help="Downscale factor for images"
    )
    parser.add_argument(
        "--sh-degree", type=int, default=3, help="Max SH degree"
    )
    parser.add_argument(
        "--lambda-dssim", type=float, default=0.2, help="SSIM loss weight"
    )
    args = parser.parse_args()

    train(
        args.scene,
        args.output,
        model_idx=args.model_idx,
        iterations=args.iterations,
        resolution_scale=args.resolution_scale,
        sh_degree=args.sh_degree,
        lambda_dssim=args.lambda_dssim,
    )
