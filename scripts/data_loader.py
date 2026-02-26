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
