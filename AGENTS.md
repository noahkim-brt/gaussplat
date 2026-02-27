# AGENTS.md

## Environment

- **Hardware**: 2x NVIDIA RTX A5000 (24GB VRAM), CUDA 12.6
- **GPU Compute Capability**: 8.6 (Ampere)
- **Python**: 3.10
- **OS**: Ubuntu Linux

## Setup

```bash
pip install -r requirements.txt
```

Set CUDA arch to avoid slow JIT compilation:
```bash
export TORCH_CUDA_ARCH_LIST="8.6"
```

## Key Commands

### End-to-end pipeline (single video)
```bash
cd scripts && python run_pipeline.py --video ../data/video.mp4
```

### End-to-end pipeline (multi-camera)
```bash
cd scripts && python run_pipeline.py --video ../data/cameras/
```

### Individual steps

**Frame extraction:**
```bash
python scripts/extract_frames.py --input data/video.mp4 --output data/scene/images --fps 2
```

**Multi-cam extraction:**
```bash
python scripts/extract_frames.py --input data/cameras/ --output data/scene --multicam
```

**SfM (Structure-from-Motion):**
```bash
python scripts/run_sfm.py --scene data/scene
python scripts/run_sfm.py --scene data/scene --matcher exhaustive --max-features 32768
```

**Training:**
```bash
cd scripts && TORCH_CUDA_ARCH_LIST="8.6" python train.py --scene ../data/scene --output ../output/scene --iterations 30000 --resolution-scale 2
```

**Resume training:**
```bash
cd scripts && python train.py --scene ../data/scene --output ../output/scene --iterations 30000 --resume ../output/scene/point_cloud/iteration_7000/checkpoint.pt
```

**View results:**
```bash
python scripts/viewer.py --ply output/scene/point_cloud/iteration_30000/point_cloud.ply
```
Then open http://localhost:8080

## Training Notes

- First run JIT-compiles gsplat CUDA kernels (~5 min). Set `TORCH_CUDA_ARCH_LIST` to speed this up.
- `--resolution-scale 2` halves images (e.g., 4K → 1080p) — much faster, still good quality.
- Training at resolution-scale 2 with 39 views: ~55 min for 30k iterations on RTX A5000.
- Densification runs from iter 500-15000. Gaussian count stabilizes after 15k.
- Checkpoints saved at iterations 7000 and 30000 by default (both PLY and .pt).

## Data Layout

```
data/<scene>/
  images/              # extracted frames
  sparse/0/            # COLMAP reconstruction (best model)
  sparse/1/            # alternative model (if exists)
  database.db          # COLMAP feature database
```

## Evaluation

Test evaluation runs every 5000 iterations by default (every 8th frame held out).
Metrics: PSNR, SSIM, LPIPS (if installed). Logged to TensorBoard in output/<scene>/tb/.

View TensorBoard:
```bash
tensorboard --logdir output/scene/tb
```
