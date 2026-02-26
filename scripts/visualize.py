"""Visualize point clouds and Gaussian splat outputs."""

import argparse

import numpy as np
import open3d as o3d
from plyfile import PlyData


def load_ply_as_pointcloud(ply_path: str) -> o3d.geometry.PointCloud:
    """Load a PLY file and return an Open3D point cloud.

    Args:
        ply_path: Path to .ply file.

    Returns:
        Open3D PointCloud object.
    """
    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]

    xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Use SH DC term for color if available
    names = vertex.data.dtype.names
    if "f_dc_0" in names:
        C0 = 0.28209479177387814
        r = vertex["f_dc_0"] * C0 + 0.5
        g = vertex["f_dc_1"] * C0 + 0.5
        b = vertex["f_dc_2"] * C0 + 0.5
        colors = np.clip(np.column_stack([r, g, b]), 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif "red" in names:
        colors = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_ply(ply_path: str) -> None:
    """Open an interactive 3D viewer for a PLY file.

    Args:
        ply_path: Path to .ply file.
    """
    print(f"Loading {ply_path}...")
    pcd = load_ply_as_pointcloud(ply_path)
    print(f"  {len(pcd.points)} points")
    o3d.visualization.draw_geometries([pcd], window_name="Gaussian Splat Viewer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PLY point cloud")
    parser.add_argument("--ply", required=True, help="Path to PLY file")
    args = parser.parse_args()

    visualize_ply(args.ply)
