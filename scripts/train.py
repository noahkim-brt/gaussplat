"""Train 3D Gaussian Splatting model using gsplat."""

import argparse
from pathlib import Path

# Placeholder — training logic will be implemented once data pipeline is validated.
# Will use gsplat rasterizer, COLMAP sparse output as initialization.


def train(scene_dir: str, output_dir: str, iterations: int = 30_000) -> None:
    """Train a Gaussian splatting model.

    Args:
        scene_dir: Scene directory with sparse/ and images/ subdirs.
        output_dir: Directory to save trained model.
        iterations: Number of training iterations.
    """
    scene_path = Path(scene_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sparse_path = scene_path / "sparse" / "0"
    images_path = scene_path / "images"

    assert sparse_path.exists(), f"Sparse reconstruction not found: {sparse_path}"
    assert images_path.exists(), f"Images not found: {images_path}"

    print(f"Scene: {scene_path}")
    print(f"Output: {output_path}")
    print(f"Iterations: {iterations}")
    print("\nTODO: Implement training loop with gsplat rasterizer")
    print("  - Load COLMAP cameras + points3D")
    print("  - Initialize Gaussians from sparse point cloud")
    print("  - Training loop: render → loss → backward → densify/prune → step")
    print("  - Save point_cloud.ply at checkpoints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Gaussian Splatting")
    parser.add_argument("--scene", required=True, help="Scene directory")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument(
        "--iterations", type=int, default=30_000, help="Training iterations"
    )
    args = parser.parse_args()

    train(args.scene, args.output, args.iterations)
