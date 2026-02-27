"""Run Structure-from-Motion using pycolmap."""

import argparse
import shutil
from pathlib import Path

import pycolmap


def run_sfm(
    scene_dir: str,
    image_dir: str = "images",
    camera_model: str = "OPENCV",
    matcher: str = "sequential",
    max_image_size: int = 3200,
    max_num_features: int = 16384,
    single_camera: bool = True,
    overlap: int = 20,
) -> Path:
    """Run the full COLMAP SfM pipeline via pycolmap.

    Args:
        scene_dir: Root directory of the scene (contains images/ subdir).
        image_dir: Name of images subdirectory within scene_dir.
        camera_model: Camera model for feature extraction.
        matcher: Matching strategy: 'sequential', 'exhaustive', or 'vocab_tree'.
        max_image_size: Max image dimension for feature extraction.
        max_num_features: Max SIFT features per image (higher = more matches, slower).
        single_camera: Assume all images share one camera (better for video).
        overlap: Number of neighboring frames to match in sequential mode.

    Returns:
        Path to the sparse reconstruction output.
    """
    scene_path = Path(scene_dir)
    images_path = scene_path / image_dir
    database_path = scene_path / "database.db"
    sparse_path = scene_path / "sparse"

    if database_path.exists():
        database_path.unlink()
        print(f"Removed existing database: {database_path}")
    if sparse_path.exists():
        shutil.rmtree(sparse_path)
        print(f"Removed existing sparse dir: {sparse_path}")
    sparse_path.mkdir(parents=True, exist_ok=True)

    num_images = len(list(images_path.glob("*.jpg")) + list(images_path.glob("*.png")))
    print(f"Scene: {scene_path}")
    print(f"Images: {images_path} ({num_images} files)")
    print(f"Matcher: {matcher}, Features: {max_num_features}, Max size: {max_image_size}")

    camera_mode = (
        pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.AUTO
    )

    sift_opts = pycolmap.SiftExtractionOptions()
    sift_opts.max_num_features = max_num_features
    sift_opts.peak_threshold = 0.004
    sift_opts.edge_threshold = 16.0

    extraction_opts = pycolmap.FeatureExtractionOptions()
    extraction_opts.max_image_size = max_image_size
    extraction_opts.use_gpu = True
    extraction_opts.sift = sift_opts

    print("\n[1/3] Feature extraction...")
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(images_path),
        camera_model=camera_model,
        camera_mode=camera_mode,
        extraction_options=extraction_opts,
    )

    print(f"\n[2/3] Feature matching ({matcher})...")
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(database_path=str(database_path))
    elif matcher == "vocab_tree":
        pycolmap.match_vocabtree(database_path=str(database_path))
    else:
        seq_opts = pycolmap.SequentialPairingOptions()
        seq_opts.overlap = overlap
        seq_opts.quadratic_overlap = True
        seq_opts.loop_detection = num_images > 50
        pycolmap.match_sequential(
            database_path=str(database_path),
            pairing_options=seq_opts,
        )

    print("\n[3/3] Sparse reconstruction (mapper)...")
    mapper_opts = pycolmap.IncrementalPipelineOptions()
    mapper_opts.min_model_size = 3
    mapper_opts.ba_refine_focal_length = True
    mapper_opts.ba_refine_extra_params = True
    mapper_opts.multiple_models = True

    maps = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(images_path),
        output_path=str(sparse_path),
        options=mapper_opts,
    )

    if maps:
        print(f"\nReconstruction complete: {len(maps)} model(s)")
        best_idx, best_images = -1, 0
        for idx, reconstruction in maps.items():
            num_reg = reconstruction.num_reg_images()
            num_pts = reconstruction.num_points3D()
            mean_reproj = reconstruction.compute_mean_reprojection_error()
            print(f"  Model {idx}: {num_reg}/{num_images} images, {num_pts} points, reproj err={mean_reproj:.3f}px")
            if num_reg > best_images:
                best_images = num_reg
                best_idx = idx
        if best_idx >= 0:
            pct = best_images / num_images * 100
            print(f"\nBest model: {best_idx} ({best_images} images, {pct:.0f}% registered)")
    else:
        print("\nWARNING: Reconstruction failed — no models produced")
        print("Try: --matcher exhaustive, --max-features 32768, or extract more frames")

    return sparse_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP SfM via pycolmap")
    parser.add_argument("--scene", required=True, help="Scene directory")
    parser.add_argument("--images", default="images", help="Image subdirectory name")
    parser.add_argument("--camera-model", default="OPENCV", help="Camera model")
    parser.add_argument(
        "--matcher",
        choices=["sequential", "exhaustive", "vocab_tree"],
        default="sequential",
        help="Matching strategy",
    )
    parser.add_argument(
        "--max-image-size", type=int, default=3200, help="Max image dimension"
    )
    parser.add_argument(
        "--max-features", type=int, default=16384, help="Max SIFT features per image"
    )
    parser.add_argument(
        "--no-single-camera",
        action="store_true",
        help="Don't assume single camera (use for multi-camera setups)",
    )
    parser.add_argument(
        "--overlap", type=int, default=20, help="Sequential matcher overlap"
    )
    args = parser.parse_args()

    run_sfm(
        args.scene,
        args.images,
        args.camera_model,
        matcher=args.matcher,
        max_image_size=args.max_image_size,
        max_num_features=args.max_features,
        single_camera=not args.no_single_camera,
        overlap=args.overlap,
    )
