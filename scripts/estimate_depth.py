"""Monocular depth estimation using Depth Anything v2 and optionally Metric3D v2.

Produces per-image depth maps saved as .npy files alongside the images.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def load_depth_anything_v2(model_size: str = "Large", device: str = "cuda"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = f"depth-anything/Depth-Anything-V2-{model_size}-hf"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return processor, model


def predict_depth_anything_v2(
    image_paths: list[Path],
    output_dir: Path,
    model_size: str = "Large",
    device: str = "cuda",
    batch_size: int = 4,
) -> list[Path]:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from PIL import Image
    from tqdm import tqdm

    model_id = f"depth-anything/Depth-Anything-V2-{model_size}-hf"
    print(f"Loading Depth Anything v2 ({model_size})...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    depth_paths = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Depth estimation"):
        batch_paths = image_paths[i : i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        orig_sizes = [(img.height, img.width) for img in images]

        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depths = outputs.predicted_depth

        for j, (path, (orig_h, orig_w)) in enumerate(zip(batch_paths, orig_sizes)):
            depth = predicted_depths[j].cpu().numpy()
            depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            out_path = output_dir / f"{path.stem}_depth.npy"
            np.save(str(out_path), depth.astype(np.float32))
            depth_paths.append(out_path)

    print(f"Saved {len(depth_paths)} depth maps to {output_dir}")
    return depth_paths


def align_depth_scale_shift(
    depth_mono: np.ndarray,
    depth_metric_sparse: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Compute scale and shift to align mono depth to metric depth via robust least squares.

    d_metric = scale * d_mono + shift

    Args:
        depth_mono: (H, W) monocular depth map (relative).
        depth_metric_sparse: (H, W) sparse metric depth (0 where unknown).
        mask: (H, W) bool, True where metric depth is valid.

    Returns:
        (scale, shift) such that scale * depth_mono + shift ≈ depth_metric_sparse.
    """
    d_m = depth_mono[mask].flatten()
    d_gt = depth_metric_sparse[mask].flatten()

    if len(d_m) < 3:
        return 1.0, 0.0

    A = np.stack([d_m, np.ones_like(d_m)], axis=-1)
    result = np.linalg.lstsq(A, d_gt, rcond=None)
    scale, shift = result[0]

    if scale <= 0:
        median_ratio = np.median(d_gt / np.clip(d_m, 1e-8, None))
        scale = median_ratio
        shift = 0.0

    return float(scale), float(shift)


def unproject_depth_to_points(
    depth: np.ndarray,
    K: np.ndarray,
    world_to_cam: np.ndarray,
    image: np.ndarray | None = None,
    stride: int = 1,
    min_depth: float = 0.1,
    max_depth: float = 500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a depth map into a 3D point cloud in world coordinates.

    Args:
        depth: (H, W) depth map.
        K: (3, 3) intrinsic matrix.
        world_to_cam: (4, 4) world-to-camera transform.
        image: (H, W, 3) RGB image for coloring points.
        stride: Pixel stride for subsampling.
        min_depth: Minimum valid depth.
        max_depth: Maximum valid depth.

    Returns:
        points_xyz: (N, 3) world-space 3D points.
        points_rgb: (N, 3) colors in [0, 1] (gray if no image).
    """
    h, w = depth.shape
    cam_to_world = np.linalg.inv(world_to_cam)

    v, u = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[v, u]
    valid = (d > min_depth) & (d < max_depth) & np.isfinite(d)

    u_valid = u[valid].astype(np.float64)
    v_valid = v[valid].astype(np.float64)
    d_valid = d[valid].astype(np.float64)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_cam = (u_valid - cx) / fx * d_valid
    y_cam = (v_valid - cy) / fy * d_valid
    z_cam = d_valid

    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=-1)
    pts_world = (cam_to_world @ pts_cam.T).T[:, :3]

    if image is not None:
        colors = image[v[valid], u[valid]]
    else:
        colors = np.full((len(pts_world), 3), 0.5, dtype=np.float32)

    return pts_world.astype(np.float32), colors.astype(np.float32)


def create_dense_init_from_depth(
    cameras,
    depth_dir: str | Path,
    voxel_size: float = 0.05,
    stride: int = 4,
    max_depth: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create dense point cloud initialization from monocular depth maps.

    Args:
        cameras: List of CameraInfo objects.
        depth_dir: Directory containing <image_stem>_depth.npy files.
        voxel_size: Voxel size for downsampling (meters). 0 = no downsampling.
        stride: Pixel stride for unprojection (higher = fewer points).
        max_depth: Maximum depth to include.

    Returns:
        points_xyz: (N, 3) merged point cloud.
        points_rgb: (N, 3) merged colors.
    """
    import open3d as o3d

    depth_dir = Path(depth_dir)
    all_pts = []
    all_rgb = []

    for cam in cameras:
        stem = Path(cam.image_name).stem
        depth_path = depth_dir / f"{stem}_depth.npy"
        if not depth_path.exists():
            continue

        depth = np.load(str(depth_path))
        h_d, w_d = depth.shape
        h_i, w_i = cam.image.shape[:2]
        if (h_d, w_d) != (h_i, w_i):
            depth = cv2.resize(depth, (w_i, h_i), interpolation=cv2.INTER_LINEAR)

        pts, rgb = unproject_depth_to_points(
            depth, cam.K, cam.world_to_cam, cam.image,
            stride=stride, max_depth=max_depth,
        )
        all_pts.append(pts)
        all_rgb.append(rgb)

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    points_xyz = np.concatenate(all_pts, axis=0)
    points_rgb = np.concatenate(all_rgb, axis=0)
    print(f"Merged point cloud: {len(points_xyz)} points from {len(all_pts)} views")

    if voxel_size > 0 and len(points_xyz) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)
        pcd.colors = o3d.utility.Vector3dVector(points_rgb)
        pcd = pcd.voxel_down_sample(voxel_size)
        points_xyz = np.asarray(pcd.points, dtype=np.float32)
        points_rgb = np.asarray(pcd.colors, dtype=np.float32)
        print(f"After voxel downsampling ({voxel_size}m): {len(points_xyz)} points")

    return points_xyz, points_rgb


def load_depth_maps_for_training(
    cameras,
    depth_dir: str | Path,
    device: str = "cuda",
) -> list[torch.Tensor | None]:
    """Load precomputed depth maps as tensors aligned to each camera view.

    Returns list of (H, W) tensors, or None if depth not found for a view.
    """
    depth_dir = Path(depth_dir)
    depth_maps = []

    for cam in cameras:
        stem = Path(cam.image_name).stem
        depth_path = depth_dir / f"{stem}_depth.npy"
        if not depth_path.exists():
            depth_maps.append(None)
            continue

        depth = np.load(str(depth_path))
        h_i, w_i = cam.image.shape[:2]
        if depth.shape != (h_i, w_i):
            depth = cv2.resize(depth, (w_i, h_i), interpolation=cv2.INTER_LINEAR)

        depth_maps.append(torch.tensor(depth, dtype=torch.float32, device=device))

    loaded = sum(1 for d in depth_maps if d is not None)
    print(f"Loaded {loaded}/{len(cameras)} depth maps for training")
    return depth_maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate monocular depth maps")
    parser.add_argument("--images", required=True, help="Directory of images")
    parser.add_argument("--output", default="", help="Output directory (default: <images>/../depth)")
    parser.add_argument("--model-size", default="Large", choices=["Small", "Base", "Large"],
                        help="Depth Anything v2 model size")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    images_dir = Path(args.images)
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    output_dir = Path(args.output) if args.output else images_dir.parent / "depth"

    predict_depth_anything_v2(
        image_paths, output_dir,
        model_size=args.model_size,
        batch_size=args.batch_size,
    )
