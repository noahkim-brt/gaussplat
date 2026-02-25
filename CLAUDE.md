# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A pipeline for creating 3D Gaussian splats from time-synchronized multi-angle video. The workflow takes multiple camera feeds captured simultaneously, reconstructs camera poses via Structure-from-Motion, trains a Gaussian splatting model, and exports viewable splats.

## Pipeline Architecture

```
Video Capture (synced cameras)
    ↓
Frame Extraction (FFmpeg)
    ↓
Camera Calibration / SfM (COLMAP)
    ↓
Gaussian Splatting Training (gsplat / 3DGS)
    ↓
Export & Viewing (.ply → web viewer / SuperSplat / glTF)
```

For **static scenes**: single video → extract frames → COLMAP → train.
For **dynamic scenes** (the primary use case here): synchronized multi-cam → per-timestep calibration → 4D Gaussian training with temporal deformation.

## Key Tools & Dependencies

**System requirements:**
- CUDA Toolkit 11.8+ (12.x preferred)
- COLMAP (pre-built binary or compiled)
- FFmpeg
- Python 3.10-3.11, conda environment recommended
- CUDA Compute Capability 7.0+ (Volta+), 8GB+ VRAM (16GB recommended)
- C++ compiler (MSVC 2019+ on Windows)

**Core Python packages:**
- `torch` (with CUDA), `torchvision` — deep learning backbone
- `gsplat` — nerfstudio's CUDA rasterizer (Apache 2.0, pip-installable, 4x less VRAM than original)
- `numpy`, `opencv-python`, `Pillow`, `plyfile` — data handling
- `open3d` — point cloud visualization
- `tqdm`, `tensorboard`, `lpips` — training infrastructure
- `ninja` — JIT CUDA kernel compilation

**Alternative to gsplat:** the original Inria `diff-gaussian-rasterization` + `simple-knn` submodules (non-commercial license, higher VRAM).

**For dynamic/4D scenes:** `4DGaussians` (HexPlane temporal deformation, CVPR 2024).

## Data Layout Conventions

**Input data (COLMAP format):**
```
data/<scene>/
  input/                    # raw videos
  images/                   # extracted frames (from ffmpeg)
  sparse/0/
    cameras.bin             # camera intrinsics
    images.bin              # camera extrinsics (poses)
    points3D.bin            # sparse point cloud
```

**Multi-camera dynamic scenes:**
```
data/<scene>/
  cam01/frame_00001.jpg ... frame_NNNNN.jpg
  cam02/frame_00001.jpg ... frame_NNNNN.jpg
  ...
```
Frame numbering must correspond across cameras (same index = same moment in time).

**Training output:**
```
output/<scene>/
  point_cloud/iteration_30000/point_cloud.ply   # final model
  cameras.json
  cfg_args
```

## Common Commands

```bash
# Frame extraction from video
ffmpeg -i input.mp4 -qscale:v 2 data/<scene>/images/frame_%04d.jpg

# COLMAP SfM pipeline (sequential matcher for video input)
colmap feature_extractor --database_path data/<scene>/database.db --image_path data/<scene>/images
colmap sequential_matcher --database_path data/<scene>/database.db
colmap mapper --database_path data/<scene>/database.db --image_path data/<scene>/images --output_path data/<scene>/sparse

# If using nerfstudio (wraps COLMAP + training):
ns-process-data video --data data/<scene>/input/video.mp4 --output-dir data/<scene>
ns-train splatfacto --data data/<scene>
```

## Architecture Decision Points

| Decision | Options | Notes |
|----------|---------|-------|
| Rasterizer | gsplat (Apache 2.0, lower VRAM) vs Inria original (higher fidelity, non-commercial) | gsplat preferred for development |
| SfM | COLMAP (robust, slow) vs MASt3R/InstantSplat (fast, no COLMAP, sparse-view) | COLMAP for production; InstantSplat for <20 views |
| Static vs Dynamic | Standard 3DGS vs 4DGaussians/SyncTrack4D | Multi-cam sync → 4DGaussians; unsync → SyncTrack4D |
| Surface extraction | SuGaR or 2DGS if mesh output needed | Only if downstream needs meshes, not splats |
| Output format | PLY (universal) vs SPZ (90% smaller) vs glTF KHR_gaussian_splatting | PLY for training/debug, glTF/SPZ for distribution |

## Camera Synchronization

This project's core challenge is multi-camera time sync. Key constraints:
- Hardware sync (genlock/trigger) is ideal; software sync (NTP) introduces sub-frame jitter
- Calibrate intrinsics/extrinsics from a static calibration frame or checkerboard pass before dynamic capture
- If cameras are unsynchronized, SyncTrack4D can learn sub-frame alignment (<0.26 frame temporal error)

## Training Parameters

Default 3DGS training: 30,000 iterations, densification from iter 500–15,000, L1 + SSIM loss. ~12 min on RTX 4090, ~25 min on RTX 3080. Output is a `.ply` file of Gaussian primitives (position, covariance, opacity, spherical harmonics color).
