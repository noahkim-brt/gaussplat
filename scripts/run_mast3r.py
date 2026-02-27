"""Run MASt3R sparse global alignment to replace COLMAP SfM.

Produces camera poses + sparse 3D points from images without COLMAP.
Works well with as few as 2-20 views.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

MAST3R_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "mast3r"
sys.path.insert(0, str(MAST3R_ROOT))
sys.path.insert(0, str(MAST3R_ROOT / "dust3r"))
sys.path.insert(0, str(MAST3R_ROOT / "dust3r" / "croco"))

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs

from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy


MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"


def run_mast3r(
    scene_dir: str,
    image_dir: str = "images",
    image_size: int = 512,
    scene_graph: str = "swin-4",
    device: str = "cuda",
    lr1: float = 0.07,
    niter1: int = 300,
    lr2: float = 0.01,
    niter2: int = 300,
    shared_intrinsics: bool = False,
    matching_conf_thr: float = 5.0,
    subsample: int = 8,
    max_images: int = 0,
) -> dict:
    """Run MASt3R reconstruction on a scene directory.

    Args:
        scene_dir: Scene directory containing images/ subdirectory.
        image_dir: Subdirectory name for images.
        image_size: Image size for MASt3R inference (512 recommended).
        scene_graph: Pair generation strategy ('complete', 'swin-N', 'logwin-N', 'oneref').
        device: Torch device.
        lr1: Coarse alignment learning rate.
        niter1: Coarse alignment iterations.
        lr2: Fine alignment learning rate.
        niter2: Fine alignment iterations.
        shared_intrinsics: Assume all images share the same camera intrinsics.
        matching_conf_thr: Confidence threshold for matching.
        subsample: Subsample factor for point maps.
        max_images: Max images to use (0 = all, useful for large datasets).

    Returns:
        Dict with 'intrinsics', 'world_to_cam', 'image_names', 'widths', 'heights',
        'points_xyz', 'points_rgb'.
    """
    scene_path = Path(scene_dir)
    images_path = scene_path / image_dir

    image_files = sorted(
        [f for f in images_path.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    if max_images > 0 and len(image_files) > max_images:
        step = len(image_files) / max_images
        image_files = [image_files[int(i * step)] for i in range(max_images)]
    print(f"Found {len(image_files)} images in {images_path}")

    file_list = [str(f) for f in image_files]

    print("Loading MASt3R model...")
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(device)

    print(f"Loading images at size={image_size}...")
    imgs = load_images(file_list, size=image_size, verbose=True)

    print(f"Making pairs with scene_graph={scene_graph}...")
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    print(f"  {len(pairs)} image pairs")

    cache_dir = str(scene_path / "mast3r_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    print("Running sparse global alignment...")
    scene = sparse_global_alignment(
        file_list, pairs, cache_dir, model,
        lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2,
        device=device, shared_intrinsics=shared_intrinsics,
        matching_conf_thr=matching_conf_thr, subsample=subsample,
    )

    intrinsics = to_numpy(scene.intrinsics)
    cam2w = to_numpy(scene.get_im_poses())
    focals = to_numpy(scene.get_focals())
    sparse_pts3d = scene.get_sparse_pts3d()

    n_imgs = len(file_list)
    all_pts_xyz = []
    all_pts_rgb = []
    for i in range(n_imgs):
        pts = to_numpy(sparse_pts3d[i])
        colors = scene.pts3d_colors[i]
        if isinstance(colors, torch.Tensor):
            colors = to_numpy(colors)
        valid = np.isfinite(pts.sum(axis=-1))
        all_pts_xyz.append(pts[valid])
        all_pts_rgb.append(colors[valid])

    if all_pts_xyz:
        points_xyz = np.concatenate(all_pts_xyz, axis=0).astype(np.float32)
        points_rgb = np.concatenate(all_pts_rgb, axis=0).astype(np.float32)
    else:
        points_xyz = np.zeros((0, 3), dtype=np.float32)
        points_rgb = np.zeros((0, 3), dtype=np.float32)

    points_rgb = np.clip(points_rgb, 0, 1)

    world_to_cam = np.zeros((n_imgs, 4, 4), dtype=np.float32)
    Ks = np.zeros((n_imgs, 3, 3), dtype=np.float32)
    image_names = []
    widths = []
    heights = []

    for i in range(n_imgs):
        c2w = cam2w[i]
        w2c = np.linalg.inv(c2w)
        world_to_cam[i] = w2c

        K = intrinsics[i]
        Ks[i] = K

        img = scene.imgs[i]
        h, w = img.shape[:2]

        image_names.append(image_files[i].name)
        widths.append(w)
        heights.append(h)

    print(f"\nMASt3R reconstruction complete:")
    print(f"  Cameras: {n_imgs}")
    print(f"  Points: {len(points_xyz)}")
    print(f"  Image size (MASt3R): {widths[0]}x{heights[0]}")

    result = {
        "intrinsics": Ks,
        "world_to_cam": world_to_cam,
        "image_names": image_names,
        "widths": np.array(widths, dtype=np.int32),
        "heights": np.array(heights, dtype=np.int32),
        "points_xyz": points_xyz,
        "points_rgb": points_rgb,
    }

    out_path = scene_path / "mast3r"
    out_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path / "reconstruction.npz"),
        **result,
    )
    print(f"  Saved to {out_path / 'reconstruction.npz'}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MASt3R reconstruction (replaces COLMAP)")
    parser.add_argument("--scene", required=True, help="Scene directory with images/ subdirectory")
    parser.add_argument("--image-dir", default="images", help="Image subdirectory name")
    parser.add_argument("--image-size", type=int, default=512, help="Image size for MASt3R (512 recommended)")
    parser.add_argument(
        "--scene-graph", default="swin-4",
        help="Pair strategy: 'complete', 'swin-N', 'logwin-N', 'oneref'",
    )
    parser.add_argument("--lr1", type=float, default=0.07, help="Coarse LR")
    parser.add_argument("--niter1", type=int, default=300, help="Coarse iterations")
    parser.add_argument("--lr2", type=float, default=0.01, help="Fine LR")
    parser.add_argument("--niter2", type=int, default=300, help="Fine iterations")
    parser.add_argument("--shared-intrinsics", action="store_true", help="Assume shared intrinsics")
    parser.add_argument("--matching-conf-thr", type=float, default=5.0, help="Matching confidence threshold")
    parser.add_argument("--max-images", type=int, default=0, help="Max images (0=all)")
    args = parser.parse_args()

    run_mast3r(
        args.scene,
        image_dir=args.image_dir,
        image_size=args.image_size,
        scene_graph=args.scene_graph,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        shared_intrinsics=args.shared_intrinsics,
        matching_conf_thr=args.matching_conf_thr,
        max_images=args.max_images,
    )
