"""Extract frames from video using FFmpeg."""

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Output directory for frames")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second")
    parser.add_argument(
        "--quality", type=int, default=2, help="JPEG quality (1=best, 31=worst)"
    )
    args = parser.parse_args()

    extract_frames(args.input, args.output, args.fps, args.quality)
