"""Load calibrated multi-camera rig with GPS/INS odometry.

Parses system_config.json (camera calibration) and starfire.json (vehicle telemetry)
to produce camera poses in world frame without any SfM.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def parse_system_config(config_path: str) -> dict:
    """Parse system_config.json for camera calibration data.

    Returns dict with:
        cameras: list of dicts with 'name', 'K', 'dist_coeffs', 'cam_from_vehicle' (4x4)
        stereo_pairs: list of (left_idx, right_idx, pair_name)
    """
    with open(config_path) as f:
        cfg = json.load(f)

    cam_names = [c["name"] for c in cfg["data"]["cameras"]]
    factory_cals = cfg["data"]["camera_factory_calibration"]

    cameras = []
    for i, cal in enumerate(factory_cals):
        if not cal.get("valid", False):
            continue

        geo = cal["geometric_calibration"]
        k_flat = geo["intrinsics"]["k"]
        K = np.array(k_flat, dtype=np.float64).reshape(3, 3)

        d = geo["intrinsics"]["d"]
        dist_coeffs = np.array(d, dtype=np.float64)

        r = geo["extrinsics"]["r"]
        t = geo["extrinsics"]["t"]
        q_xyzw = [r["x"], r["y"], r["z"], r["w"]]
        R_cv = Rotation.from_quat(q_xyzw).as_matrix()
        t_cv = np.array([t["x"], t["y"], t["z"]], dtype=np.float64)

        cam_from_vehicle = np.eye(4, dtype=np.float64)
        cam_from_vehicle[:3, :3] = R_cv
        cam_from_vehicle[:3, 3] = t_cv

        cameras.append({
            "name": cam_names[i] if i < len(cam_names) else f"CAM{i:02d}",
            "K": K,
            "dist_coeffs": dist_coeffs,
            "cam_from_vehicle": cam_from_vehicle,
        })

    stereo_pairs = []
    for sp in cfg["data"].get("stereo_pairs", []):
        stereo_pairs.append((
            sp["left_camera_index"],
            sp["right_camera_index"],
            sp["pair_name"],
        ))

    return {"cameras": cameras, "stereo_pairs": stereo_pairs}


def parse_starfire(starfire_path: str) -> list[dict]:
    """Parse starfire.json for vehicle telemetry.

    Returns list of dicts (primary GPS only) with:
        timestamp_ns, lat, lon, alt, bearing, pitch, roll, speed
    """
    with open(starfire_path) as f:
        data = json.load(f)

    records = []
    for entry in data:
        d = entry["data"]
        if d.get("type") != "primary":
            continue
        records.append({
            "timestamp_ns": d["timestamp"],
            "lat": d["latitude"],
            "lon": d["longitude"],
            "alt": d["altitude"],
            "bearing": d["bearing"],
            "pitch": d["pitch"],
            "roll": d["roll"],
            "speed": d["speed"],
        })

    records.sort(key=lambda r: r["timestamp_ns"])
    return records


def gps_to_enu(lat: float, lon: float, alt: float,
               lat0: float, lon0: float, alt0: float) -> np.ndarray:
    """Convert GPS lat/lon/alt to local ENU coordinates (meters)."""
    R_EARTH = 6371000.0
    lat0_rad = np.radians(lat0)
    east = R_EARTH * np.radians(lon - lon0) * np.cos(lat0_rad)
    north = R_EARTH * np.radians(lat - lat0)
    up = alt - alt0
    return np.array([east, north, up], dtype=np.float64)


def vehicle_pose_from_gps(bearing_deg: float, pitch_deg: float, roll_deg: float,
                          east: float, north: float, up: float) -> np.ndarray:
    """Construct world_from_vehicle 4x4 matrix from GPS orientation + position.

    Vehicle frame: X=forward, Y=left, Z=up (FLU).
    World frame: X=East, Y=North, Z=Up (ENU).

    At bearing=0 (North): vehicle X → world Y, vehicle Y → world -X.
    At bearing=90 (East): vehicle X → world X, vehicle Y → world Y.
    """
    yaw_rad = np.radians(90.0 - bearing_deg)
    pitch_rad = np.radians(pitch_deg)
    roll_rad = np.radians(roll_deg)

    R_yaw = Rotation.from_euler("z", yaw_rad)
    R_pitch = Rotation.from_euler("y", -pitch_rad)
    R_roll = Rotation.from_euler("x", roll_rad)
    R_wv = (R_yaw * R_pitch * R_roll).as_matrix()

    world_from_vehicle = np.eye(4, dtype=np.float64)
    world_from_vehicle[:3, :3] = R_wv
    world_from_vehicle[:3, 3] = [east, north, up]
    return world_from_vehicle


def interpolate_gps(records: list[dict], target_ns: int) -> dict:
    """Linearly interpolate GPS records to a target timestamp."""
    if target_ns <= records[0]["timestamp_ns"]:
        return records[0]
    if target_ns >= records[-1]["timestamp_ns"]:
        return records[-1]

    for j in range(len(records) - 1):
        t0, t1 = records[j]["timestamp_ns"], records[j + 1]["timestamp_ns"]
        if t0 <= target_ns <= t1:
            alpha = (target_ns - t0) / (t1 - t0)
            r0, r1 = records[j], records[j + 1]

            interp = {}
            for k in ("lat", "lon", "alt", "pitch", "roll", "speed"):
                interp[k] = r0[k] * (1 - alpha) + r1[k] * alpha

            b0, b1 = np.radians(r0["bearing"]), np.radians(r1["bearing"])
            db = np.arctan2(np.sin(b1 - b0), np.cos(b1 - b0))
            interp["bearing"] = np.degrees(b0 + alpha * db)

            interp["timestamp_ns"] = target_ns
            return interp

    return records[-1]


def undistort_image(image: np.ndarray, K: np.ndarray,
                    dist_coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undistort image using OpenCV rational model. Returns (undistorted, new_K)."""
    h, w = image.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha=0)
    undistorted = cv2.undistort(image, K, dist_coeffs, None, new_K)

    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y + rh, x:x + rw]
        new_K[0, 2] -= x
        new_K[1, 2] -= y

    return undistorted, new_K


def load_calibrated_rig(
    scene_dir: str,
    config_name: str = "system_config.json",
    telemetry_name: str = "starfire.json",
    image_dir: str = "images",
    camera_indices: list[int] | None = None,
    timestamp_ns_list: list[int] | None = None,
    undistort: bool = True,
    resolution_scale: int = 1,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Load a calibrated rig scene with GPS odometry.

    Expects:
        scene_dir/
            system_config.json
            starfire.json
            images/
                <CAM_NAME>_<TIMESTAMP_NS>.jpg  (or organized per camera)

    Args:
        scene_dir: Scene directory.
        config_name: Camera config filename.
        telemetry_name: GPS telemetry filename.
        image_dir: Image subdirectory.
        camera_indices: Which cameras to use (None = all).
        timestamp_ns_list: Which timestamps to use (None = auto from images).
        undistort: Whether to undistort images.
        resolution_scale: Downscale factor.

    Returns:
        cameras: List of CameraInfo-compatible dicts.
        points_xyz: Empty (N, 3) array (no initial points from calibration alone).
        points_rgb: Empty (N, 3) array.
    """
    from data_loader import CameraInfo

    scene_path = Path(scene_dir)
    rig = parse_system_config(str(scene_path / config_name))
    gps_records = parse_starfire(str(scene_path / telemetry_name))

    if not gps_records:
        raise ValueError("No primary GPS records found in starfire.json")

    ref = gps_records[0]
    lat0, lon0, alt0 = ref["lat"], ref["lon"], ref["alt"]

    images_path = scene_path / image_dir
    if camera_indices is None:
        camera_indices = list(range(len(rig["cameras"])))

    image_files = sorted(images_path.glob("*.jpg")) + sorted(images_path.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_path}")

    cam_lookup = {cam["name"]: idx for idx, cam in enumerate(rig["cameras"])}

    cameras_out = []
    used_timestamps = set()

    for img_path in image_files:
        stem = img_path.stem
        parts = stem.split("_", 1)
        if len(parts) == 2:
            cam_name, ts_str = parts
        else:
            cam_name = stem
            ts_str = None

        if cam_name not in cam_lookup:
            continue
        cam_idx = cam_lookup[cam_name]
        if cam_idx not in camera_indices:
            continue

        cam = rig["cameras"][cam_idx]

        if ts_str is not None:
            ts_ns = int(ts_str)
        elif timestamp_ns_list:
            ts_ns = timestamp_ns_list[0]
        else:
            ts_ns = gps_records[len(gps_records) // 2]["timestamp_ns"]

        if timestamp_ns_list and ts_ns not in timestamp_ns_list:
            continue

        gps = interpolate_gps(gps_records, ts_ns)
        enu = gps_to_enu(gps["lat"], gps["lon"], gps["alt"], lat0, lon0, alt0)
        world_from_vehicle = vehicle_pose_from_gps(
            gps["bearing"], gps["pitch"], gps["roll"],
            enu[0], enu[1], enu[2],
        )

        vehicle_from_world = np.linalg.inv(world_from_vehicle)
        world_to_cam = cam["cam_from_vehicle"] @ vehicle_from_world

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        K = cam["K"].copy()
        if undistort:
            img, K = undistort_image(img, K, cam["dist_coeffs"])

        h, w = img.shape[:2]
        if resolution_scale > 1:
            new_w, new_h = w // resolution_scale, h // resolution_scale
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            K[0, :] /= resolution_scale
            K[1, :] /= resolution_scale
            w, h = new_w, new_h

        img = img.astype(np.float32) / 255.0

        cameras_out.append(CameraInfo(
            image_name=img_path.name,
            width=w,
            height=h,
            K=K.astype(np.float32),
            world_to_cam=world_to_cam.astype(np.float32),
            image=img,
        ))
        used_timestamps.add(ts_ns)

    print(f"Loaded {len(cameras_out)} views from calibrated rig")
    print(f"  Cameras: {len(set(camera_indices))} | Timestamps: {len(used_timestamps)}")

    points_xyz = np.zeros((0, 3), dtype=np.float32)
    points_rgb = np.zeros((0, 3), dtype=np.float32)

    return cameras_out, points_xyz, points_rgb


def generate_initial_points(cameras_out, rig: dict, gps_records: list,
                            lat0: float, lon0: float, alt0: float,
                            num_depth_samples: int = 50,
                            max_depth: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate initial 3D points by ray-casting from camera centers.

    Since we have no SfM points, we create a sparse set by sampling rays
    from each camera at various depths.
    """
    all_pts = []
    all_rgb = []

    for cam_info in cameras_out:
        w2c = cam_info.world_to_cam
        c2w = np.linalg.inv(w2c)
        cam_center = c2w[:3, 3]
        K = cam_info.K

        h, w = cam_info.image.shape[:2]
        img = cam_info.image

        sample_u = np.linspace(w * 0.1, w * 0.9, 8).astype(int)
        sample_v = np.linspace(h * 0.1, h * 0.9, 6).astype(int)
        depths = np.linspace(5.0, max_depth, num_depth_samples)

        K_inv = np.linalg.inv(K.astype(np.float64))

        for u in sample_u:
            for v in sample_v:
                pixel = np.array([u, v, 1.0])
                ray_cam = K_inv @ pixel
                ray_cam /= np.linalg.norm(ray_cam)
                ray_world = c2w[:3, :3] @ ray_cam

                color = img[min(v, h - 1), min(u, w - 1)]

                for d in depths:
                    pt = cam_center + d * ray_world
                    all_pts.append(pt)
                    all_rgb.append(color)

    points_xyz = np.array(all_pts, dtype=np.float32)
    points_rgb = np.array(all_rgb, dtype=np.float32)

    return points_xyz, points_rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load calibrated rig + GPS odometry")
    parser.add_argument("--scene", required=True, help="Scene directory")
    parser.add_argument("--config", default="system_config.json")
    parser.add_argument("--telemetry", default="starfire.json")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--no-undistort", action="store_true")
    args = parser.parse_args()

    cameras, pts_xyz, pts_rgb = load_calibrated_rig(
        args.scene,
        config_name=args.config,
        telemetry_name=args.telemetry,
        image_dir=args.image_dir,
        undistort=not args.no_undistort,
    )
    print(f"\nResult: {len(cameras)} camera views, {len(pts_xyz)} points")
    for c in cameras[:5]:
        print(f"  {c.image_name}: {c.width}x{c.height}")
