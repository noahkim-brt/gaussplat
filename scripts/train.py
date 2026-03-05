"""Train 3D Gaussian Splatting model using gsplat.

Supports depth-regularized training following DNGaussian (Li et al., CVPR 2024):
  - Hard depth loss: L1 between rendered depth and mono depth (with per-image scale-shift)
  - Soft depth loss: Pearson correlation (scale-shift invariant)
  - Dense initialization from unprojected depth maps (FSGS-style)
"""

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

from data_loader import (
    cameras_to_tensors,
    compute_scene_scale,
    load_colmap_scene,
    load_mast3r_scene,
    split_train_test,
)
from load_calibrated_rig import load_calibrated_rig, generate_initial_points
from gaussian_model import init_gaussians, load_checkpoint, save_checkpoint, save_ply
from estimate_depth import (
    create_dense_init_from_depth,
    load_depth_maps_for_training,
    predict_depth_anything_v2,
)

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


# ─── Depth losses ────────────────────────────────────────────────────────────


def depth_loss_hard(
    rendered_depth: torch.Tensor,
    mono_depth: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Hard depth loss: L1 between rendered depth and scale-shift aligned mono depth.

    Following DNGaussian: d_aligned = scale * d_mono + shift
    """
    aligned = scale * mono_depth + shift
    valid = (aligned > 0.01) & (rendered_depth > 0.01)
    if valid.sum() < 10:
        return torch.tensor(0.0, device=rendered_depth.device)
    return F.l1_loss(rendered_depth[valid], aligned[valid])


def depth_loss_pearson(
    rendered_depth: torch.Tensor,
    mono_depth: torch.Tensor,
) -> torch.Tensor:
    """Soft depth loss: 1 - Pearson correlation (scale-shift invariant)."""
    valid = (mono_depth > 0.01) & (rendered_depth > 0.01)
    if valid.sum() < 10:
        return torch.tensor(0.0, device=rendered_depth.device)
    r = rendered_depth[valid]
    m = mono_depth[valid]
    r_centered = r - r.mean()
    m_centered = m - m.mean()
    corr = (r_centered * m_centered).sum() / (
        torch.sqrt((r_centered**2).sum() * (m_centered**2).sum()) + 1e-8
    )
    return 1.0 - corr


def compute_optimal_scale_shift(
    rendered_depth: torch.Tensor,
    mono_depth: torch.Tensor,
) -> tuple[float, float]:
    """Analytically compute optimal scale/shift (MonoSDF approach)."""
    valid = (mono_depth > 0.01) & (rendered_depth > 0.01)
    if valid.sum() < 10:
        return 1.0, 0.0
    r = rendered_depth[valid].detach()
    m = mono_depth[valid]
    m_mean, r_mean = m.mean(), r.mean()
    m_centered = m - m_mean
    cov = (m_centered * (r - r_mean)).sum()
    var = (m_centered**2).sum() + 1e-8
    scale = cov / var
    shift = r_mean - scale * m_mean
    return scale.item(), shift.item()


# ─── Metrics ─────────────────────────────────────────────────────────────────


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    mse = F.mse_loss(img1, img2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def evaluate(
    params: dict,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    gt_images: list[torch.Tensor],
    widths: list[int],
    heights: list[int],
    sh_degree: int = 3,
    lpips_fn=None,
) -> dict[str, float]:
    """Evaluate on a set of views. Returns dict of averaged metrics."""
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    colors = torch.cat([params["sh0"], params["sh_rest"]], dim=1)

    for i in range(len(gt_images)):
        rendered, _, _ = rasterization(
            means=params["means"],
            quats=params["quats"],
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=colors,
            viewmats=viewmats[i : i + 1],
            Ks=Ks[i : i + 1],
            width=widths[i],
            height=heights[i],
            sh_degree=sh_degree,
            packed=True,
            render_mode="RGB",
        )
        rendered_image = rendered[0]
        gt = gt_images[i]

        psnr_vals.append(psnr(rendered_image, gt))
        ssim_vals.append(_ssim(rendered_image, gt).item())

        if lpips_fn is not None:
            r = rendered_image.permute(2, 0, 1).unsqueeze(0) * 2 - 1
            g = gt.permute(2, 0, 1).unsqueeze(0) * 2 - 1
            lpips_vals.append(lpips_fn(r, g).item())

    metrics = {
        "psnr": sum(psnr_vals) / len(psnr_vals),
        "ssim": sum(ssim_vals) / len(ssim_vals),
    }
    if lpips_vals:
        metrics["lpips"] = sum(lpips_vals) / len(lpips_vals)
    return metrics


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
    test_interval: int = 5_000,
    test_every_n: int = 8,
    resume_from: str = "",
    sfm_method: str = "colmap",
    use_depth: bool = False,
    depth_model_size: str = "Large",
    lambda_depth_hard: float = 0.5,
    lambda_depth_pearson: float = 0.1,
    depth_loss_from: int = 0,
    depth_loss_until: int = 15_000,
    depth_init: bool = False,
    depth_init_voxel: float = 0.05,
    depth_init_stride: int = 8,
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
        test_interval: Run evaluation every N iterations (0 = disabled).
        test_every_n: Hold out every N-th view for testing.
        resume_from: Path to checkpoint .pt file to resume training from.
        sfm_method: SfM method used ('colmap' or 'mast3r').
        use_depth: Enable depth-regularized training.
        depth_model_size: Depth Anything v2 model size.
        lambda_depth_hard: Weight for hard depth loss (L1).
        lambda_depth_pearson: Weight for soft depth loss (Pearson).
        depth_loss_from: Start depth loss at this iteration.
        depth_loss_until: Stop depth loss at this iteration (0 = never stop).
        depth_init: Use dense depth-unprojected point cloud for initialization.
        depth_init_voxel: Voxel size for depth init downsampling.
        depth_init_stride: Pixel stride for depth unprojection.
    """
    if save_iterations is None:
        save_iterations = [7_000, 30_000]

    device = "cuda"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print(f"Loading scene data (sfm={sfm_method})...")
    if sfm_method == "calibrated_rig":
        all_cameras, points_xyz, points_rgb = load_calibrated_rig(
            scene_dir, resolution_scale=resolution_scale
        )
        if len(points_xyz) == 0:
            print("No SfM points — generating initial points from camera rays...")
            points_xyz, points_rgb = generate_initial_points(all_cameras, None, None, 0, 0, 0)
            print(f"  Generated {len(points_xyz)} initial points")
    elif sfm_method == "mast3r":
        all_cameras, points_xyz, points_rgb = load_mast3r_scene(
            scene_dir, resolution_scale=resolution_scale
        )
    else:
        all_cameras, points_xyz, points_rgb = load_colmap_scene(
            scene_dir, model_idx=model_idx, resolution_scale=resolution_scale
        )
    scene_scale = compute_scene_scale(points_xyz)

    # ── Depth estimation ──────────────────────────────────────────────────
    depth_dir = Path(scene_dir) / "depth"
    depth_maps_train = None
    if use_depth:
        images_dir = Path(scene_dir) / "images"
        existing = list(depth_dir.glob("*_depth.npy"))
        if len(existing) < len(all_cameras):
            print("Running Depth Anything v2 estimation...")
            image_paths = sorted(
                [images_dir / cam.image_name for cam in all_cameras]
            )
            predict_depth_anything_v2(
                image_paths, depth_dir,
                model_size=depth_model_size,
                batch_size=4,
            )
            torch.cuda.empty_cache()
        else:
            print(f"Found {len(existing)} existing depth maps in {depth_dir}")

        if depth_init and not resume_from:
            print("Creating dense initialization from depth maps...")
            dense_xyz, dense_rgb = create_dense_init_from_depth(
                all_cameras, depth_dir,
                voxel_size=depth_init_voxel,
                stride=depth_init_stride,
            )
            if len(dense_xyz) > 0:
                import numpy as np
                points_xyz = np.concatenate([points_xyz, dense_xyz], axis=0)
                points_rgb = np.concatenate([points_rgb, dense_rgb], axis=0)
                print(f"Combined init: {len(points_xyz)} points (SfM + depth)")
                scene_scale = compute_scene_scale(points_xyz)

    if test_interval > 0 and len(all_cameras) > 4:
        train_cameras, test_cameras = split_train_test(all_cameras, every_n=test_every_n)
    else:
        train_cameras, test_cameras = all_cameras, []

    viewmats, Ks, gt_images, widths, heights = cameras_to_tensors(train_cameras, device)
    num_views = len(train_cameras)
    print(f"Scene scale: {scene_scale:.4f}")
    print(f"Image resolution: {widths[0]}x{heights[0]}")

    if use_depth:
        depth_maps_train = load_depth_maps_for_training(train_cameras, depth_dir, device)
        depth_scales = [None] * num_views
        depth_shifts = [None] * num_views

    if test_cameras:
        test_viewmats, test_Ks, test_gt_images, test_widths, test_heights = (
            cameras_to_tensors(test_cameras, device)
        )
        try:
            import lpips as lpips_lib
            lpips_fn = lpips_lib.LPIPS(net="vgg").to(device)
            print("LPIPS loaded (vgg)")
        except ImportError:
            lpips_fn = None
            print("LPIPS not available, skipping")
    else:
        test_viewmats = test_Ks = test_gt_images = test_widths = test_heights = None
        lpips_fn = None

    # ── Initialize Gaussians ──────────────────────────────────────────────
    start_step = 0
    if resume_from:
        params, optimizer_states, start_step = load_checkpoint(resume_from, device)
    else:
        params = init_gaussians(points_xyz, points_rgb, device)
        optimizer_states = None

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

    if optimizer_states:
        for k, opt in optimizers.items():
            if k in optimizer_states:
                opt.load_state_dict(optimizer_states[k])

    # ── Densification strategy ────────────────────────────────────────────
    strategy = DefaultStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    # ── Tensorboard ───────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(output_path / "tb"))

    # ── Training loop ─────────────────────────────────────────────────────
    remaining = iterations - start_step
    depth_active = use_depth and depth_maps_train is not None
    render_mode = "RGB+ED" if depth_active else "RGB"

    print(f"\nStarting training for {iterations} iterations (resuming from {start_step})...")
    print(f"  Views: {num_views}")
    print(f"  Gaussians: {params['means'].shape[0]}")
    print(f"  Loss: (1-{lambda_dssim})*L1 + {lambda_dssim}*SSIM")
    if depth_active:
        print(f"  Depth: hard={lambda_depth_hard} pearson={lambda_depth_pearson} (iters {depth_loss_from}-{depth_loss_until})")

    pbar = tqdm(range(start_step + 1, iterations + 1), desc="Training", initial=start_step, total=iterations)
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
            render_mode=render_mode,
        )

        if depth_active:
            rendered_image = rendered[0, :, :, :3]  # (H, W, 3)
            rendered_depth = rendered[0, :, :, 3]   # (H, W) expected depth
        else:
            rendered_image = rendered[0]  # (H, W, 3)
            rendered_depth = None

        # Loss: L1 + SSIM
        l1_loss = F.l1_loss(rendered_image, gt_image)
        ssim_loss = 1.0 - _ssim(rendered_image, gt_image)
        loss = (1.0 - lambda_dssim) * l1_loss + lambda_dssim * ssim_loss

        # Depth losses
        d_hard_loss = torch.tensor(0.0, device=device)
        d_pearson_loss = torch.tensor(0.0, device=device)
        if (
            depth_active
            and depth_maps_train[idx] is not None
            and step >= depth_loss_from
            and (depth_loss_until == 0 or step <= depth_loss_until)
        ):
            mono_depth = depth_maps_train[idx]

            if depth_scales[idx] is None or step % 500 == 0:
                s, sh = compute_optimal_scale_shift(rendered_depth.detach(), mono_depth)
                depth_scales[idx] = s
                depth_shifts[idx] = sh

            scale_t = torch.tensor(depth_scales[idx], device=device)
            shift_t = torch.tensor(depth_shifts[idx], device=device)

            if lambda_depth_hard > 0:
                d_hard_loss = depth_loss_hard(rendered_depth, mono_depth, scale_t, shift_t)
                loss = loss + lambda_depth_hard * d_hard_loss

            if lambda_depth_pearson > 0:
                d_pearson_loss = depth_loss_pearson(rendered_depth, mono_depth)
                loss = loss + lambda_depth_pearson * d_pearson_loss

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
            if depth_active:
                writer.add_scalar("train/depth_hard_loss", d_hard_loss.item(), step)
                writer.add_scalar("train/depth_pearson_loss", d_pearson_loss.item(), step)
            running_loss = 0.0

        # Save checkpoints
        if step in save_iterations:
            ckpt_dir = output_path / "point_cloud" / f"iteration_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_ply(params, str(ckpt_dir / "point_cloud.ply"))
            save_checkpoint(params, optimizers, step, str(ckpt_dir / "checkpoint.pt"))

        # Test evaluation
        if test_cameras and test_interval > 0 and step % test_interval == 0:
            metrics = evaluate(
                params, test_viewmats, test_Ks, test_gt_images,
                test_widths, test_heights, sh_degree, lpips_fn,
            )
            msg = f"[Test @ {step}] PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f}"
            if "lpips" in metrics:
                msg += f" LPIPS={metrics['lpips']:.4f}"
            tqdm.write(msg)
            writer.add_scalar("test/psnr", metrics["psnr"], step)
            writer.add_scalar("test/ssim", metrics["ssim"], step)
            if "lpips" in metrics:
                writer.add_scalar("test/lpips", metrics["lpips"], step)

    # ── Final save ────────────────────────────────────────────────────────
    total_time = time.time() - t0
    final_dir = output_path / "point_cloud" / f"iteration_{iterations}"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_ply(params, str(final_dir / "point_cloud.ply"))
    save_checkpoint(params, optimizers, iterations, str(final_dir / "checkpoint.pt"))

    # ── Final evaluation ──────────────────────────────────────────────────
    if test_cameras:
        print("\nFinal evaluation on test views...")
        final_metrics = evaluate(
            params, test_viewmats, test_Ks, test_gt_images,
            test_widths, test_heights, sh_degree, lpips_fn,
        )
        print(f"  PSNR:  {final_metrics['psnr']:.2f} dB")
        print(f"  SSIM:  {final_metrics['ssim']:.4f}")
        if "lpips" in final_metrics:
            print(f"  LPIPS: {final_metrics['lpips']:.4f}")
        for k, v in final_metrics.items():
            writer.add_scalar(f"final/{k}", v, iterations)

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
    parser.add_argument(
        "--test-interval", type=int, default=5000, help="Test eval interval (0=disable)"
    )
    parser.add_argument(
        "--test-every-n", type=int, default=8, help="Hold out every N-th view for testing"
    )
    parser.add_argument(
        "--resume", default="", help="Path to checkpoint.pt to resume training from"
    )
    parser.add_argument(
        "--sfm-method", choices=["colmap", "mast3r", "calibrated_rig"], default="colmap",
        help="SfM method used for reconstruction",
    )
    parser.add_argument("--use-depth", action="store_true", help="Enable depth-regularized training")
    parser.add_argument("--depth-model-size", default="Large", choices=["Small", "Base", "Large"],
                        help="Depth Anything v2 model size")
    parser.add_argument("--lambda-depth-hard", type=float, default=0.5, help="Hard depth loss weight")
    parser.add_argument("--lambda-depth-pearson", type=float, default=0.1, help="Pearson depth loss weight")
    parser.add_argument("--depth-loss-from", type=int, default=0, help="Start depth loss at iteration")
    parser.add_argument("--depth-loss-until", type=int, default=15000,
                        help="Stop depth loss at iteration (0=never)")
    parser.add_argument("--depth-init", action="store_true",
                        help="Initialize with dense depth-unprojected point cloud")
    parser.add_argument("--depth-init-voxel", type=float, default=0.05,
                        help="Voxel size for depth init downsampling")
    parser.add_argument("--depth-init-stride", type=int, default=8,
                        help="Pixel stride for depth unprojection")
    args = parser.parse_args()

    train(
        args.scene,
        args.output,
        model_idx=args.model_idx,
        iterations=args.iterations,
        resolution_scale=args.resolution_scale,
        sh_degree=args.sh_degree,
        lambda_dssim=args.lambda_dssim,
        test_interval=args.test_interval,
        test_every_n=args.test_every_n,
        resume_from=args.resume,
        sfm_method=args.sfm_method,
        use_depth=args.use_depth,
        depth_model_size=args.depth_model_size,
        lambda_depth_hard=args.lambda_depth_hard,
        lambda_depth_pearson=args.lambda_depth_pearson,
        depth_loss_from=args.depth_loss_from,
        depth_loss_until=args.depth_loss_until,
        depth_init=args.depth_init,
        depth_init_voxel=args.depth_init_voxel,
        depth_init_stride=args.depth_init_stride,
    )
