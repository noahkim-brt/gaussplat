"""End-to-end pipeline: video → frames → SfM → train → export."""

import argparse
from pathlib import Path

from extract_frames import extract_frames, extract_multicam
from run_sfm import run_sfm
from train import train

MAST3R_AVAILABLE = False
try:
    from run_mast3r import run_mast3r
    MAST3R_AVAILABLE = True
except ImportError:
    pass


def run_pipeline(
    video_path: str,
    scene_name: str = "",
    data_root: str = "data",
    output_root: str = "output",
    fps: int = 2,
    quality: int = 2,
    matcher: str = "sequential",
    max_features: int = 16384,
    max_image_size: int = 3200,
    iterations: int = 30_000,
    resolution_scale: int = 1,
    sh_degree: int = 3,
    test_interval: int = 5_000,
    sfm_method: str = "colmap",
    mast3r_image_size: int = 512,
    mast3r_scene_graph: str = "swin-4",
    mast3r_max_images: int = 0,
) -> None:
    """Run the full Gaussian splatting pipeline from video to trained model.

    Args:
        video_path: Path to input video file.
        scene_name: Scene name (defaults to video filename stem).
        data_root: Root directory for scene data.
        output_root: Root directory for training output.
        fps: Frames per second to extract.
        quality: JPEG quality for frame extraction (1=best, 31=worst).
        matcher: COLMAP matcher strategy.
        max_features: Max SIFT features per image.
        max_image_size: Max image dimension for feature extraction.
        iterations: Training iterations.
        resolution_scale: Downscale factor for training images.
        sh_degree: Max spherical harmonics degree.
        test_interval: Evaluation interval (0 to disable).
        sfm_method: SfM method ('colmap' or 'mast3r').
        mast3r_image_size: Image size for MASt3R inference.
        mast3r_scene_graph: MASt3R pair generation strategy.
        mast3r_max_images: Max images for MASt3R (0 = all).
    """
    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    multicam = input_path.is_dir()

    if not scene_name:
        scene_name = input_path.stem if not multicam else input_path.name

    scene_dir = Path(data_root) / scene_name
    output_dir = Path(output_root) / scene_name
    images_dir = scene_dir / "images"

    mode = "multi-camera" if multicam else "single video"
    print(f"{'='*60}")
    print(f"Pipeline ({mode}): {input_path.name} → {scene_name}")
    print(f"  Scene dir:  {scene_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}")

    # Step 1: Extract frames
    print(f"\n{'─'*60}")
    print("STEP 1: Frame extraction")
    print(f"{'─'*60}")
    if images_dir.exists() and any(images_dir.glob("*.jpg")):
        num_existing = len(list(images_dir.glob("*.jpg")))
        print(f"Skipping: {num_existing} frames already exist in {images_dir}")
    elif multicam:
        extract_multicam(str(input_path), str(scene_dir), fps=fps, quality=quality)
    else:
        extract_frames(str(input_path), str(images_dir), fps=fps, quality=quality)

    # Step 2: SfM
    print(f"\n{'─'*60}")
    print(f"STEP 2: Structure-from-Motion ({sfm_method})")
    print(f"{'─'*60}")
    if sfm_method == "mast3r":
        if not MAST3R_AVAILABLE:
            raise RuntimeError(
                "MASt3R not available. Ensure third_party/mast3r is cloned "
                "with submodules and dependencies are installed."
            )
        mast3r_recon = scene_dir / "mast3r" / "reconstruction.npz"
        if mast3r_recon.exists():
            print(f"Skipping: MASt3R reconstruction already exists at {mast3r_recon}")
        else:
            run_mast3r(
                str(scene_dir),
                image_size=mast3r_image_size,
                scene_graph=mast3r_scene_graph,
                max_images=mast3r_max_images,
                shared_intrinsics=not multicam,
            )
    else:
        sparse_dir = scene_dir / "sparse"
        if sparse_dir.exists() and any(sparse_dir.glob("*/cameras.bin")):
            num_models = len(list(sparse_dir.glob("*/cameras.bin")))
            print(f"Skipping: {num_models} model(s) already exist in {sparse_dir}")
        else:
            run_sfm(
                str(scene_dir),
                matcher="exhaustive" if multicam else matcher,
                max_num_features=max_features,
                max_image_size=max_image_size,
                single_camera=not multicam,
            )

    # Step 3: Train
    print(f"\n{'─'*60}")
    print("STEP 3: Gaussian Splatting Training")
    print(f"{'─'*60}")
    train(
        str(scene_dir),
        str(output_dir),
        iterations=iterations,
        resolution_scale=resolution_scale,
        sh_degree=sh_degree,
        test_interval=test_interval,
        sfm_method=sfm_method,
    )

    # Summary
    final_ply = output_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"  Output PLY: {final_ply}")
    print(f"  View: python scripts/viewer.py --ply {final_ply}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end: video → frames → SfM → Gaussian splatting"
    )
    parser.add_argument("--video", required=True, help="Video file or directory of videos (auto-detects multi-cam)")
    parser.add_argument("--scene-name", default="", help="Scene name (default: video stem)")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    parser.add_argument("--output-root", default="output", help="Output root directory")
    parser.add_argument("--fps", type=int, default=2, help="Frame extraction FPS")
    parser.add_argument("--quality", type=int, default=2, help="JPEG quality (1=best)")
    parser.add_argument(
        "--matcher", choices=["sequential", "exhaustive", "vocab_tree"],
        default="sequential", help="COLMAP matcher",
    )
    parser.add_argument("--max-features", type=int, default=16384, help="SIFT features per image")
    parser.add_argument("--max-image-size", type=int, default=3200, help="Max image dimension")
    parser.add_argument("--iterations", type=int, default=30_000, help="Training iterations")
    parser.add_argument("--resolution-scale", type=int, default=1, help="Image downscale factor")
    parser.add_argument("--sh-degree", type=int, default=3, help="Max SH degree")
    parser.add_argument("--test-interval", type=int, default=5000, help="Test eval interval")
    parser.add_argument(
        "--sfm-method", choices=["colmap", "mast3r"], default="colmap",
        help="SfM method: 'colmap' (default) or 'mast3r' (no COLMAP needed, works with few views)",
    )
    parser.add_argument("--mast3r-image-size", type=int, default=512, help="MASt3R inference image size")
    parser.add_argument("--mast3r-scene-graph", default="swin-4", help="MASt3R pair strategy")
    parser.add_argument("--mast3r-max-images", type=int, default=0, help="Max images for MASt3R (0=all)")
    args = parser.parse_args()

    run_pipeline(
        args.video,
        scene_name=args.scene_name,
        data_root=args.data_root,
        output_root=args.output_root,
        fps=args.fps,
        quality=args.quality,
        matcher=args.matcher,
        max_features=args.max_features,
        max_image_size=args.max_image_size,
        iterations=args.iterations,
        resolution_scale=args.resolution_scale,
        sh_degree=args.sh_degree,
        test_interval=args.test_interval,
        sfm_method=args.sfm_method,
        mast3r_image_size=args.mast3r_image_size,
        mast3r_scene_graph=args.mast3r_scene_graph,
        mast3r_max_images=args.mast3r_max_images,
    )
