# gaussplat

Pipeline for creating 3D Gaussian splats from time-synchronized multi-angle video.

## Pipeline

```
Video Capture (synced cameras)
    ↓
Frame Extraction (FFmpeg)
    ↓
Camera Calibration / SfM (COLMAP via pycolmap)
    ↓
Gaussian Splatting Training (gsplat)
    ↓
Export & Viewing (.ply)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires CUDA 11.8+ and FFmpeg.

## Usage

### 1. Extract frames from video

```bash
python scripts/extract_frames.py --input data/scene/input/video.mp4 --output data/scene/images --fps 2
```

### 2. Run Structure-from-Motion

```bash
python scripts/run_sfm.py --scene data/scene
```

### 3. Train Gaussian splatting model

```bash
python scripts/train.py --scene data/scene --output output/scene
```

### 4. Visualize output

```bash
python scripts/visualize.py --ply output/scene/point_cloud.ply
```

## Project Structure

```
gaussplat/
  data/                  # Input scenes and raw video
  output/                # Training outputs (.ply models)
  scripts/
    extract_frames.py    # FFmpeg frame extraction
    run_sfm.py           # COLMAP SfM via pycolmap
    train.py             # Gaussian splatting training
    export.py            # Export model to PLY
    visualize.py         # Open3D point cloud viewer
  configs/
    default.py           # Training and pipeline configs
  requirements.txt
```

## Hardware

- 2x NVIDIA RTX A5000 (24GB VRAM)
- CUDA 12.6
