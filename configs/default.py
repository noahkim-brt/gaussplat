"""Default training configuration for 3D Gaussian Splatting."""

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # Data
    scene_dir: str = "data/scene"
    output_dir: str = "output"
    image_dir: str = "images"
    resolution: int = -1  # -1 = use original resolution

    # Training
    iterations: int = 30_000
    lr_position: float = 1.6e-4
    lr_opacity: float = 0.05
    lr_scaling: float = 0.005
    lr_rotation: float = 0.001
    lr_sh: float = 0.0025
    lr_sh_rest: float = 0.000125

    # Densification
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_interval: int = 100
    densify_grad_threshold: float = 0.0002
    opacity_reset_interval: int = 3_000
    min_opacity: float = 0.005

    # Loss
    lambda_dssim: float = 0.2  # weight for SSIM loss (1 - lambda = L1 weight)

    # Logging
    save_iterations: list = field(default_factory=lambda: [7_000, 30_000])
    log_interval: int = 100
    test_interval: int = 1_000

    # SH degree
    sh_degree: int = 3


@dataclass
class FrameExtractionConfig:
    input_video: str = ""
    output_dir: str = ""
    fps: int = 2  # frames per second to extract
    quality: int = 2  # ffmpeg qscale (1=best, 31=worst)


@dataclass
class SfMConfig:
    scene_dir: str = ""
    image_dir: str = "images"
    camera_model: str = "OPENCV"
    use_sequential_matcher: bool = True  # True for video input, False for unordered photos
    max_image_size: int = 3200
