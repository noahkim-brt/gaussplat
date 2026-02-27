"""Load COLMAP reconstruction data for Gaussian splatting training."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch


@dataclass
class CameraInfo:
    """Camera intrinsics and extrinsics for a single view."""

    image_name: str
    width: int
    height: int
    K: np.ndarray  # (3, 3) intrinsic matrix
    world_to_cam: np.ndarray  # (4, 4) extrinsic matrix
    image: np.ndarray  # (H, W, 3) RGB float32 [0, 1]


def load_colmap_scene(
    scene_dir: str,
    model_idx: int = 0,
    image_dir: str = "images",
    resolution_scale: int = 1,
) -> tuple[list[CameraInfo], np.ndarray, np.ndarray]:
    """Load a COLMAP sparse reconstruction and associated images.

    Args:
        scene_dir: Root scene directory containing sparse/ and images/.
        model_idx: Which COLMAP model to load (0, 1, ...).
        image_dir: Subdirectory name for images.
        resolution_scale: Downscale factor for images (1 = original).

    Returns:
        cameras: List of CameraInfo for each registered image.
        points_xyz: (N, 3) initial 3D point positions.
        points_rgb: (N, 3) initial point colors in [0, 1].
    """
    scene_path = Path(scene_dir)
    sparse_path = scene_path / "sparse" / str(model_idx)
    images_path = scene_path / image_dir

    reconstruction = pycolmap.Reconstruction(str(sparse_path))

    # Load 3D points
    points_xyz_list = []
    points_rgb_list = []
    for point in reconstruction.points3D.values():
        points_xyz_list.append(point.xyz)
        points_rgb_list.append(point.color / 255.0)

    points_xyz = np.array(points_xyz_list, dtype=np.float32)
    points_rgb = np.array(points_rgb_list, dtype=np.float32)

    print(f"Loaded {len(points_xyz)} sparse 3D points")

    # Load cameras and images
    cameras = []
    for image_id, image in reconstruction.images.items():
        if not image.has_pose:
            continue

        cam = reconstruction.cameras[image.camera_id]

        # Intrinsics
        K = np.array(cam.calibration_matrix(), dtype=np.float32)

        # Extrinsics: cam_from_world -> 4x4 matrix
        rigid = image.cam_from_world()
        w2c_3x4 = np.array(rigid.matrix(), dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = w2c_3x4

        # Load image
        img_path = images_path / image.name
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Downscale if needed
        if resolution_scale > 1:
            new_w = cam.width // resolution_scale
            new_h = cam.height // resolution_scale
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            K[0, :] /= resolution_scale
            K[1, :] /= resolution_scale
            width, height = new_w, new_h
        else:
            width, height = cam.width, cam.height

        img = img.astype(np.float32) / 255.0

        cameras.append(
            CameraInfo(
                image_name=image.name,
                width=width,
                height=height,
                K=K,
                world_to_cam=w2c,
                image=img,
            )
        )

    print(f"Loaded {len(cameras)} registered camera views")
    return cameras, points_xyz, points_rgb


def cameras_to_tensors(
    cameras: list[CameraInfo], device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[int], list[int]]:
    """Convert camera list to tensors for gsplat rasterization.

    Args:
        cameras: List of CameraInfo objects.
        device: Torch device.

    Returns:
        viewmats: (N, 4, 4) world-to-camera matrices.
        Ks: (N, 3, 3) intrinsic matrices.
        images: List of (H, W, 3) image tensors.
        widths: List of image widths.
        heights: List of image heights.
    """
    viewmats = torch.tensor(
        np.stack([c.world_to_cam for c in cameras]), dtype=torch.float32, device=device
    )
    Ks = torch.tensor(
        np.stack([c.K for c in cameras]), dtype=torch.float32, device=device
    )
    images = [
        torch.tensor(c.image, dtype=torch.float32, device=device) for c in cameras
    ]
    widths = [c.width for c in cameras]
    heights = [c.height for c in cameras]

    return viewmats, Ks, images, widths, heights


def split_train_test(
    cameras: list[CameraInfo],
    test_ratio: float = 0.125,
    every_n: int | None = None,
) -> tuple[list[CameraInfo], list[CameraInfo]]:
    """Split cameras into train and test sets.

    Args:
        cameras: All camera views, sorted by name.
        test_ratio: Fraction of views to hold out for testing.
        every_n: If set, pick every N-th view as test (overrides test_ratio).

    Returns:
        (train_cameras, test_cameras)
    """
    sorted_cams = sorted(cameras, key=lambda c: c.image_name)
    if every_n is None:
        every_n = max(1, round(1.0 / test_ratio))

    train, test = [], []
    for i, cam in enumerate(sorted_cams):
        if i % every_n == 0:
            test.append(cam)
        else:
            train.append(cam)

    print(f"Split: {len(train)} train, {len(test)} test (every {every_n}th frame)")
    return train, test


def load_mast3r_scene(
    scene_dir: str,
    image_dir: str = "images",
    resolution_scale: int = 1,
) -> tuple[list[CameraInfo], np.ndarray, np.ndarray]:
    """Load a MASt3R reconstruction and associated images.

    MASt3R saves camera intrinsics at its inference resolution (e.g. 512px).
    We load the original full-resolution images and rescale K accordingly.

    Args:
        scene_dir: Root scene directory containing mast3r/ and images/.
        image_dir: Subdirectory name for images.
        resolution_scale: Downscale factor for images (1 = original).

    Returns:
        cameras: List of CameraInfo for each reconstructed view.
        points_xyz: (N, 3) initial 3D point positions.
        points_rgb: (N, 3) initial point colors in [0, 1].
    """
    scene_path = Path(scene_dir)
    recon_path = scene_path / "mast3r" / "reconstruction.npz"
    images_path = scene_path / image_dir

    data = np.load(str(recon_path), allow_pickle=True)
    Ks = data["intrinsics"]
    w2cs = data["world_to_cam"]
    image_names = data["image_names"]
    mast3r_widths = data["widths"]
    mast3r_heights = data["heights"]
    points_xyz = data["points_xyz"]
    points_rgb = data["points_rgb"]

    print(f"Loaded {len(points_xyz)} sparse 3D points (MASt3R)")

    cameras = []
    for i, name in enumerate(image_names):
        img_path = images_path / name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: could not load {img_path}, skipping")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        K = Ks[i].copy()
        sx = orig_w / mast3r_widths[i]
        sy = orig_h / mast3r_heights[i]
        K[0, :] *= sx
        K[1, :] *= sy

        if resolution_scale > 1:
            new_w = orig_w // resolution_scale
            new_h = orig_h // resolution_scale
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            K[0, :] /= resolution_scale
            K[1, :] /= resolution_scale
            width, height = new_w, new_h
        else:
            width, height = orig_w, orig_h

        img = img.astype(np.float32) / 255.0

        cameras.append(
            CameraInfo(
                image_name=str(name),
                width=width,
                height=height,
                K=K,
                world_to_cam=w2cs[i],
                image=img,
            )
        )

    print(f"Loaded {len(cameras)} camera views (MASt3R)")
    return cameras, points_xyz, points_rgb


def compute_scene_scale(points_xyz: np.ndarray) -> float:
    """Compute scene scale as the average distance from centroid.

    Args:
        points_xyz: (N, 3) point positions.

    Returns:
        Scene scale (average distance from centroid).
    """
    centroid = points_xyz.mean(axis=0)
    distances = np.linalg.norm(points_xyz - centroid, axis=1)
    return float(np.mean(distances))
