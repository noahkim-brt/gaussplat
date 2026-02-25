"""Run Structure-from-Motion using pycolmap."""

import argparse
from pathlib import Path

import pycolmap


def run_sfm(
    scene_dir: str,
    image_dir: str = "images",
    camera_model: str = "OPENCV",
    sequential: bool = True,
    max_image_size: int = 3200,
) -> Path:
    """Run the full COLMAP SfM pipeline via pycolmap.

    Args:
        scene_dir: Root directory of the scene (contains images/ subdir).
        image_dir: Name of images subdirectory within scene_dir.
        camera_model: Camera model for feature extraction.
        sequential: Use sequential matcher (True for video) or exhaustive (False).
        max_image_size: Max image dimension for feature extraction.

    Returns:
        Path to the sparse reconstruction output.
    """
    scene_path = Path(scene_dir)
    images_path = scene_path / image_dir
    database_path = scene_path / "database.db"
    sparse_path = scene_path / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)

    print(f"Scene: {scene_path}")
    print(f"Images: {images_path}")
    print(f"Database: {database_path}")

    # Feature extraction
    print("\n[1/3] Feature extraction...")
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(images_path),
        camera_model=camera_model,
        sift_options=pycolmap.SiftExtractionOptions(
            max_image_size=max_image_size,
        ),
    )

    # Feature matching
    print("\n[2/3] Feature matching...")
    if sequential:
        print("  Using sequential matcher (video input)")
        pycolmap.match_sequential(database_path=str(database_path))
    else:
        print("  Using exhaustive matcher (unordered photos)")
        pycolmap.match_exhaustive(database_path=str(database_path))

    # Sparse reconstruction
    print("\n[3/3] Sparse reconstruction (mapper)...")
    maps = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(images_path),
        output_path=str(sparse_path),
    )

    if maps:
        print(f"\n  Reconstruction complete: {len(maps)} model(s)")
        for idx, reconstruction in maps.items():
            num_images = reconstruction.num_reg_images()
            num_points = reconstruction.num_points3D()
            print(f"  Model {idx}: {num_images} images, {num_points} points")
    else:
        print("\n  WARNING: Reconstruction failed — no models produced")

    return sparse_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP SfM via pycolmap")
    parser.add_argument("--scene", required=True, help="Scene directory")
    parser.add_argument("--images", default="images", help="Image subdirectory name")
    parser.add_argument("--camera-model", default="OPENCV", help="Camera model")
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Use exhaustive matcher instead of sequential",
    )
    parser.add_argument(
        "--max-image-size", type=int, default=3200, help="Max image dimension"
    )
    args = parser.parse_args()

    run_sfm(
        args.scene,
        args.images,
        args.camera_model,
        sequential=not args.exhaustive,
        max_image_size=args.max_image_size,
    )
