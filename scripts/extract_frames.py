"""Extract frames from video using FFmpeg.

Supports single video and multi-camera setups.
"""

import argparse
import subprocess
from pathlib import Path


def extract_frames(
    input_video: str,
    output_dir: str,
    fps: int = 2,
    quality: int = 2,
) -> Path:
    """Extract frames from a video file.

    Args:
        input_video: Path to input video file.
        output_dir: Directory to write extracted frames.
        fps: Frames per second to extract.
        quality: FFmpeg qscale value (1=best, 31=worst).

    Returns:
        Path to the output directory containing frames.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame_pattern = str(output_path / "frame_%05d.jpg")

    cmd = [
        "ffmpeg",
        "-i",
        input_video,
        "-vf",
        f"fps={fps}",
        "-qscale:v",
        str(quality),
        frame_pattern,
    ]

    print(f"Extracting frames: {input_video} -> {output_dir}")
    print(f"  FPS: {fps}, Quality: {quality}")
    subprocess.run(cmd, check=True)

    frame_count = len(list(output_path.glob("frame_*.jpg")))
    print(f"  Extracted {frame_count} frames")

    return output_path


def extract_multicam(
    video_dir: str,
    scene_dir: str,
    fps: int = 2,
    quality: int = 2,
) -> dict[str, Path]:
    """Extract frames from multiple synchronized camera videos.

    Expects video files named like cam01.mp4, cam02.mp4, etc. in video_dir.
    Outputs to scene_dir/cam01/, scene_dir/cam02/, etc.

    Also creates a merged scene_dir/images/ directory with frames renamed as
    cam01_frame_00001.jpg, cam02_frame_00001.jpg, etc. for COLMAP.

    Args:
        video_dir: Directory containing camera video files.
        scene_dir: Scene root directory for output.
        fps: Frames per second to extract.
        quality: FFmpeg qscale value.

    Returns:
        Dict mapping camera name to output directory Path.
    """
    video_path = Path(video_dir)
    scene_path = Path(scene_dir)

    videos = sorted(
        list(video_path.glob("*.mp4"))
        + list(video_path.glob("*.MP4"))
        + list(video_path.glob("*.mov"))
        + list(video_path.glob("*.MOV"))
        + list(video_path.glob("*.avi"))
    )

    if not videos:
        raise FileNotFoundError(f"No video files found in {video_dir}")

    print(f"Found {len(videos)} camera videos in {video_dir}")
    cam_dirs = {}

    for video_file in videos:
        cam_name = video_file.stem
        cam_dir = scene_path / cam_name
        print(f"\n--- Camera: {cam_name} ---")
        extract_frames(str(video_file), str(cam_dir), fps=fps, quality=quality)
        cam_dirs[cam_name] = cam_dir

    images_dir = scene_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    frame_counts = {}
    for cam_name, cam_dir in cam_dirs.items():
        frames = sorted(cam_dir.glob("frame_*.jpg"))
        frame_counts[cam_name] = len(frames)
        for frame in frames:
            dst = images_dir / f"{cam_name}_{frame.name}"
            if not dst.exists():
                dst.symlink_to(frame.resolve())

    min_frames = min(frame_counts.values()) if frame_counts else 0
    max_frames = max(frame_counts.values()) if frame_counts else 0

    print(f"\nMulti-cam extraction complete:")
    print(f"  Cameras: {len(cam_dirs)}")
    print(f"  Frames per camera: {min_frames}-{max_frames}")
    print(f"  Merged images dir: {images_dir} ({sum(frame_counts.values())} total)")

    if min_frames != max_frames:
        print(f"  WARNING: Frame counts differ across cameras. Check sync.")

    return cam_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--input", required=True, help="Video file or directory of videos (multi-cam)")
    parser.add_argument("--output", required=True, help="Output directory for frames")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second")
    parser.add_argument(
        "--quality", type=int, default=2, help="JPEG quality (1=best, 31=worst)"
    )
    parser.add_argument(
        "--multicam", action="store_true", help="Multi-camera mode: --input is a directory of videos"
    )
    args = parser.parse_args()

    if args.multicam:
        extract_multicam(args.input, args.output, args.fps, args.quality)
    else:
        extract_frames(args.input, args.output, args.fps, args.quality)
