# Literature Review: Monocular Depth-Guided 3D Scene Reconstruction via Gaussian Splatting and Neural Radiance Fields

## Abstract

This review surveys methods for reconstructing 3D scenes—particularly as 3D Gaussian Splats (3DGS) and Neural Radiance Fields (NeRFs)—using monocular depth estimation as a geometric prior, given calibrated cameras with known intrinsics and extrinsics. We cover: (1) foundations of monocular depth estimation, (2) depth-supervised NeRFs, (3) depth-regularized Gaussian splatting, (4) feed-forward 3D prediction models, (5) depth unprojection pipelines, and (6) the scale alignment problem. We conclude with practical recommendations for a calibrated multi-camera rig in an outdoor mining/quarry environment.

---

## 1. Monocular Depth Estimation Foundations

Monocular depth estimation (MDE) predicts per-pixel depth from a single RGB image. The field has seen transformative progress from 2019-2024, with a critical distinction between **relative/affine-invariant depth** and **metric depth** models.

### 1.1 Relative / Affine-Invariant Depth Models

These models predict depth up to an unknown scale and shift: d_metric = s * d_predicted + t. They are trained on diverse datasets with heterogeneous depth representations, achieving remarkable zero-shot generalization but **not** producing metric-scale output.

#### MiDaS (Ranftl et al., 2020, TPAMI / arXiv 2019)

- **Key contribution**: Pioneered multi-dataset training for robust monocular depth. Introduced a scale-and-shift-invariant loss that allowed mixing datasets with different depth scales (stereo, SfM, laser, etc.). The resulting model generalized far better to unseen domains than any single-dataset model.
- **Architecture**: Initially used encoder-decoder CNNs (EfficientNet, ResNeXt-101). Later versions (MiDaS v3, v3.1) incorporated Vision Transformers.
- **Output**: Inverse depth (disparity), affine-invariant. No camera intrinsics required at inference.
- **Relevance**: Foundational work; all subsequent affine-invariant models build on its multi-dataset training paradigm. The inverse depth output requires careful scale-shift alignment when used with metric SfM points (see S6).

#### DPT -- Dense Prediction Transformer (Ranftl et al., ICCV 2021)

- **Key contribution**: Replaced CNN backbones with Vision Transformers (ViT) for dense prediction. Showed that ViT features, when reassembled into multi-scale feature maps via "reassemble" and "fusion" blocks, significantly outperformed CNN-based MiDaS on zero-shot transfer.
- **Architecture**: ViT-Base/Large backbone -> Reassemble (project ViT tokens to spatial maps at multiple resolutions) -> Fusion (progressive upsampling with residual connections) -> prediction head.
- **Output**: Affine-invariant inverse depth. No intrinsics needed.
- **Relevance**: DPT became the de facto backbone for MiDaS v3.x. Its architecture directly inspired Depth Anything.

#### Depth Anything v1 (Yang, Kang, et al., CVPR 2024)

- **Key contribution**: Scaled up training data massively (~62M unlabeled images via self-training) while using a DPT-style architecture with DINOv2 backbones. Employed a teacher-student framework: a teacher trained on labeled data generates pseudo-labels for unlabeled data; a student is trained on both, with strong augmentations on unlabeled data to learn robust features.
- **Architecture**: DINOv2 (ViT-S/B/L) encoder + DPT decoder.
- **Output**: Affine-invariant relative depth. No intrinsics needed.
- **Key insight**: Data scaling + self-training on unlabeled images was more impactful than architecture changes. The resulting model showed state-of-the-art zero-shot relative depth across diverse benchmarks.
- **Relevance**: Excellent general-purpose relative depth backbone. When used as a prior for 3DGS, requires scale-shift alignment per image.

#### Depth Anything v2 (Yang, Kang, et al., arXiv 2024 / NeurIPS 2024)

- **Key contribution**: Replaced the pseudo-labeled real images with high-quality synthetic data (595K images from photorealistic synthetic datasets including Hypersim, Virtual KITTI, and custom synthetic data). Key insight: training on precise synthetic depth then fine-tuning/mixing with limited real labeled data outperforms training on noisy pseudo-labeled real data.
- **Architecture**: Same DINOv2 + DPT architecture as v1 but with improved training recipe.
- **Output**: Affine-invariant relative depth. No intrinsics needed. Also released fine-tuned metric depth variants for specific domains.
- **Improvements over v1**: Sharper edges, better fine detail, fewer artifacts, more robust across domains.
- **Relevance**: Currently the strongest general-purpose relative depth model. The metric fine-tuned variants are relevant if the domain matches; the relative model is more robust for novel domains (e.g., mining/quarry). **Strong candidate as depth prior for our use case.**

#### Marigold (Ke et al., CVPR 2024)

- **Key contribution**: Repurposed a pre-trained latent diffusion model (Stable Diffusion) for monocular depth estimation. Fine-tuned only on synthetic data (Hypersim + Virtual KITTI) by reformulating depth prediction as a conditional image generation problem. The diffusion model generates depth maps conditioned on RGB input.
- **Architecture**: Stable Diffusion U-Net, operating in latent space. At inference, denoises a random latent into a depth map latent, conditioned on the RGB image encoded by the same VAE.
- **Output**: Affine-invariant relative depth. No intrinsics needed.
- **Key insight**: The rich visual priors in large-scale image diffusion models transfer powerfully to geometric tasks. Achieved excellent zero-shot depth despite training on only ~74K synthetic images. Produces ensembles for uncertainty estimation.
- **Limitations**: Slower than feed-forward models (requires iterative denoising, typically 10-50 steps). Ensemble inference compounds cost.
- **Relevance**: Excellent edge quality and fine detail. The diffusion-based approach is complementary to discriminative models. Uncertainty estimates could be valuable for weighting depth loss terms. However, speed may be a concern for 12 cameras x many frames.

### 1.2 Metric Depth Models

These models predict depth in absolute metric units (meters), requiring or implicitly encoding camera intrinsics.

#### ZoeDepth (Bhat et al., arXiv 2023)

- **Key contribution**: Two-stage framework: (1) a pre-trained relative depth backbone (MiDaS/BTS) provides robust features, (2) a lightweight "metric bins" module maps relative features to metric depth using learned bin centers. Introduced "ZoeD-M12-NK" trained on 12 datasets with domain-specific heads.
- **Architecture**: Relative backbone (MiDaS DPT) + metric bins module (attractor-based bin assignment).
- **Input**: RGB image only (no explicit intrinsics input, but trained on datasets with specific camera distributions).
- **Output**: Metric depth in meters.
- **Key insight**: Decoupling relative feature learning from metric depth prediction allows combining the generalization of relative models with metric accuracy.
- **Limitations**: Metric accuracy degrades on out-of-distribution cameras/scenes. Indoor/outdoor domain gap significant.
- **Relevance**: If the mining/quarry domain is sufficiently close to outdoor training data (KITTI-like), ZoeDepth can provide metric depth directly. However, industrial scenes may be out of distribution.

#### Metric3D (Yin et al., ICCV 2023) and Metric3D v2 (Hu, Yin, et al., arXiv 2024)

- **Key contribution**: Explicitly addresses the intrinsics ambiguity in metric depth. Key insight: the focal length is the primary source of metric ambiguity. Metric3D proposes a **canonical camera transformation** -- it resizes/crops the input image to simulate a canonical focal length, predicts depth for this canonical view, then rescales the output to the original camera's metric space.
- **Architecture (v1)**: Standard encoder-decoder with canonical camera normalization as preprocessing.
- **Architecture (v2)**: DINOv2 backbone, adds a learned camera embedding and more sophisticated de-canonicalization. Also jointly predicts surface normals.
- **Input**: RGB image + focal length (required for canonical camera transform).
- **Output**: Metric depth in meters.
- **Key insight**: By factoring out the focal length from the learning problem, the model can train on diverse datasets while maintaining metric accuracy. This is the correct theoretical framing: depth and focal length are fundamentally entangled in monocular images (doubling focal length and depth produces the same image).
- **Relevance**: **Highly relevant.** We have known camera intrinsics (focal length), so we can provide the required input. Metric3D can directly produce metric depth, potentially avoiding the scale-shift alignment problem entirely. The canonical camera normalization is elegant for multi-camera rigs with different focal lengths.

#### UniDepth (Piccinelli et al., CVPR 2024)

- **Key contribution**: Predicts both metric depth AND camera intrinsics from a single image in a single forward pass. Uses a self-prompting camera module that infers a camera representation (including focal length) from the image features, then conditions the depth decoder on this camera representation.
- **Architecture**: ViT encoder -> self-prompting camera module (predicts intrinsics) -> depth decoder conditioned on camera tokens.
- **Input**: RGB image only (intrinsics predicted, not required; but can be provided as input to improve accuracy).
- **Output**: Metric depth in meters + predicted camera intrinsics.
- **Key insight**: The camera module learns to infer focal length from visual cues (field of view, perspective distortion), enabling metric depth without explicit intrinsics. When intrinsics are known, they can override the prediction.
- **Relevance**: Useful as a fallback or validation tool. Since we have known intrinsics, we can provide them directly. The joint prediction is less critical for our calibrated rig but demonstrates the importance of intrinsics-awareness for metric depth.

### 1.3 Summary Table: Monocular Depth Models

| Model | Year | Output Type | Needs Intrinsics? | Architecture | Best For |
|-------|------|-------------|-------------------|-------------|----------|
| MiDaS v3.1 | 2019-2022 | Affine-invariant | No | CNN/ViT + DPT | General relative depth |
| DPT | 2021 | Affine-invariant | No | ViT + DPT | Backbone for MiDaS |
| Depth Anything v1 | 2024 | Affine-invariant | No | DINOv2 + DPT | Large-scale robust relative depth |
| Depth Anything v2 | 2024 | Affine-invariant | No | DINOv2 + DPT | SOTA relative depth, sharp edges |
| Marigold | 2024 | Affine-invariant | No | Stable Diffusion | High detail, uncertainty |
| ZoeDepth | 2023 | Metric | No (implicit) | MiDaS + bins | Metric depth (known domains) |
| Metric3D v2 | 2024 | Metric | **Yes** (focal length) | DINOv2 + canonical cam | Metric depth with known cameras |
| UniDepth | 2024 | Metric | Optional | ViT + camera module | Metric depth, intrinsics prediction |

---

## 2. Depth-Supervised Neural Radiance Fields

NeRFs (Mildenhall et al., ECCV 2020) learn a volumetric scene representation by optimizing an MLP to reproduce input images via differentiable volume rendering. They struggle with sparse views due to underconstrained geometry. Monocular depth priors provide geometric supervision to regularize NeRF training.

### 2.1 DS-NeRF -- Depth-Supervised NeRF (Deng et al., CVPR 2022)

- **Key contribution**: First work to supervise NeRF with depth from structure-from-motion (SfM). Adds a depth loss that encourages the NeRF's rendered depth to match sparse SfM point cloud depths. Uses the ray termination distribution to define a depth probability and applies a KL divergence loss.
- **Depth source**: Sparse SfM points (from COLMAP), not monocular depth.
- **Loss**: KL divergence between NeRF ray depth distribution and a Gaussian centered on SfM depth.
- **Result**: Significantly improved novel view synthesis quality with fewer input views (3-5 views).
- **Relevance**: Establishes that depth supervision helps NeRFs with sparse views. The KL loss on ray distributions is more principled than simple L2 depth loss. For our use case with 12 cameras and known poses, this sparse depth supervision approach is a starting point, though dense monocular depth provides much richer supervision.

### 2.2 MonoSDF (Yu et al., NeurIPS 2022)

- **Key contribution**: Integrates monocular depth and normal priors into neural implicit surface reconstruction (SDF-based NeRF). Uses a signed distance function instead of density, enabling better surface reconstruction. Supervises with mono depth from Omnidata and mono normals.
- **Architecture**: NeuS/VolSDF-style SDF network + monocular geometric priors.
- **Loss**: Scale-shift-invariant depth loss + angular normal loss. The depth loss optimizes per-image scale and shift parameters to align mono depth with rendered depth.
- **Key insight**: Monocular priors are especially valuable in textureless regions and for regularizing surface geometry beyond just appearance. The scale-shift invariant loss is critical for using affine-invariant mono depth.
- **Relevance**: The scale-shift-invariant depth loss formulation is directly applicable to our setting. MonoSDF shows that even with many views, mono depth priors improve surface geometry quality, especially in textureless regions common in mining environments.

### 2.3 NeuralRGBD (Azinovic et al., 3DV 2022)

- **Key contribution**: Uses depth sensor data (RGB-D) to supervise a NeRF-like neural implicit representation. Demonstrates truncated signed distance function (TSDF) integration with neural rendering. While focused on depth sensors, the framework extends to monocular depth.
- **Depth source**: Depth sensor (but applicable to monocular depth with appropriate noise modeling).
- **Key insight**: Depth supervision works best when combined with a truncation-based SDF formulation that handles depth uncertainty. Depth near object boundaries and at far ranges is less reliable and should be down-weighted.
- **Relevance**: The uncertainty-aware depth integration is relevant for outdoor scenes where monocular depth is less reliable at long ranges (common in quarry/mining).

### 2.4 RegNeRF (Niemeyer et al., CVPR 2022)

- **Key contribution**: Regularizes NeRF training from sparse views by rendering **unobserved viewpoints** and applying: (1) an appearance regularization (normalizing flow prior on rendered patches), and (2) a depth smoothness regularization on rendered depth maps from novel views.
- **Depth supervision**: Not from monocular depth models; instead uses a learned depth smoothness prior as regularization.
- **Key insight**: Regularizing geometry in unobserved viewpoints is critical for sparse-view NeRF. Without it, NeRFs can overfit to training views while producing degenerate geometry elsewhere.
- **Relevance**: Complementary to monocular depth supervision. In our multi-camera setup, there may be significant gaps between viewing angles; RegNeRF-style regularization of unobserved viewpoints could supplement mono depth losses.

### 2.5 DiffusioNeRF (Wynn & Turmukhambetov, CVPR 2023)

- **Key contribution**: Uses a denoising diffusion model trained on RGBD patches as a 3D geometry prior for NeRF. At each training iteration, renders RGBD patches from random viewpoints and passes them through a denoising step of the diffusion model, using the score (gradient of log-likelihood) as a regularization signal.
- **Key insight**: Diffusion models capture rich geometric priors about 3D scenes. Using them as a regularizer during NeRF optimization provides a data-driven prior that is more flexible than hand-crafted smoothness terms.
- **Relevance**: Demonstrates the value of generative model priors for 3D reconstruction. Related to Marigold's use of diffusion models but applied differently (as a regularizer rather than a direct predictor).

### 2.6 Summary of Depth-Supervised NeRF Approaches

The consensus from this literature is clear: **monocular depth priors significantly improve NeRF geometry, especially with sparse or widely-spaced views**. Key design choices:
- Use **scale-shift-invariant losses** when using affine-invariant depth models.
- **Per-image** scale/shift parameters are necessary (not global) because mono depth scale varies per image.
- Depth supervision is most impactful in **textureless regions** and **occluded areas**.
- Combining depth with **normal** supervision further improves surface quality.

---

## 3. Depth-Regularized 3D Gaussian Splatting

3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) represents scenes as collections of 3D Gaussian primitives, each with position (mean), covariance (shape/orientation), opacity, and spherical harmonics (color). It achieves real-time rendering via a tile-based rasterizer. Like NeRFs, 3DGS struggles with sparse views, and monocular depth priors offer a powerful regularization.

### 3.1 DNGaussian -- Depth-Regularized Optimization for Sparse-View Gaussian Splatting (Li et al., CVPR 2024)

- **Key contribution**: First dedicated work on depth-regularized 3DGS for sparse views. Proposes two key innovations:
  1. **Hard depth regularization**: Uses monocular depth to constrain Gaussian positions via a depth rendering loss. Applies a scale-shift alignment per image.
  2. **Soft depth regularization**: A depth ranking/ordinal loss that enforces *relative depth ordering* rather than absolute depth values. This is more robust to mono depth errors.
- **Depth model used**: DPT/MiDaS (affine-invariant).
- **Loss formulation**:
  - Hard: L_hard = | d_hat - (s * d_mono + t) |_1 where s, t are per-image learnable parameters.
  - Soft: Pearson correlation loss between rendered depth and mono depth, which is inherently scale-shift invariant.
- **Key insight**: **The combination of hard + soft depth losses is more effective than either alone.** Hard loss provides metric grounding; soft/ranking loss handles regions where mono depth has incorrect absolute values but correct relative ordering.
- **Results**: Significant improvements on LLFF, DTU, and Blender datasets with 3-8 views.
- **Relevance**: **Directly applicable to our use case.** The dual hard/soft loss formulation is well-suited for outdoor scenes where mono depth may have local inaccuracies. With known poses, we can directly apply this approach. The hard loss benefits from metric depth (Metric3D), while the soft loss provides robustness.

### 3.2 FSGS -- Few-Shot Gaussian Splatting (Zhu et al., CVPR 2024)

- **Key contribution**: Addresses sparse-view 3DGS via a proximity-guided Gaussian unprojection strategy. Uses monocular depth to unproject sparse SfM points into dense point clouds for better Gaussian initialization, combined with a depth-based Gaussian growing strategy.
- **Approach**:
  1. Run SfM (COLMAP) to get sparse points + camera poses.
  2. Predict monocular depth (Depth Anything or similar).
  3. Align mono depth to SfM points via scale-shift fitting.
  4. Unproject aligned mono depth to get dense initialization point clouds.
  5. During 3DGS training, use depth loss + a proximity-guided Gaussian unprojection strategy that adds new Gaussians in underrepresented regions by unprojecting from depth maps.
- **Key insight**: **Initialization quality matters enormously for 3DGS.** Starting from dense, depth-derived point clouds rather than sparse SfM points dramatically improves convergence and final quality, especially for sparse views.
- **Relevance**: **Highly relevant pipeline.** Our calibrated rig provides known poses (no SfM needed for poses, though we could still run SfM for sparse points for scale alignment). The dense initialization via depth unprojection is directly applicable.

### 3.3 SparseGS (Xiong et al., arXiv 2023)

- **Key contribution**: Combines depth priors with a floater-pruning strategy for sparse-view 3DGS. Uses monocular depth for: (1) depth ranking loss for geometric regularization, (2) a depth-guided Gaussian pruning strategy to remove "floater" artifacts (Gaussians that appear in the rendering of one view but are geometrically inconsistent with depth from another).
- **Depth loss**: Ordinal/ranking loss -- for randomly sampled pixel pairs (i, j), if d_mono(i) > d_mono(j), enforce d_hat(i) > d_hat(j). This is purely ordinal and avoids scale/shift issues entirely.
- **Key insight**: **Floater artifacts** are a major problem in sparse-view 3DGS. Depth priors help both in regularization and in identifying inconsistent Gaussians for removal.
- **Relevance**: Floater pruning is important for outdoor scenes with sky/open areas. The ordinal depth loss avoids the scale alignment problem entirely, trading absolute accuracy for robustness.

### 3.4 DepthRegularizedGS (Chung et al., CVPR 2024)

- **Key contribution**: Proposes a general framework for depth-regularized 3DGS that works with various depth sources (sensor, SfM, monocular). Introduces a **depth distortion loss** inspired by Mip-NeRF 360's distortion loss, applied to the per-Gaussian depth contributions.
- **Loss**: Combines depth rendering loss (L1/L2) with a distortion loss that encourages Gaussians along a ray to be concentrated rather than spread out, promoting clean surfaces.
- **Key insight**: The distortion loss prevents Gaussians from spreading along rays, which causes blurry rendering and degenerate geometry. This is particularly important when depth supervision is available to anchor geometry.
- **Relevance**: The distortion loss is a valuable addition to any depth-supervised 3DGS pipeline. Applicable to our setting.

### 3.5 Key Insight: Depth Ranking/Ordinal Loss vs. Absolute Depth Loss

This is a critical design choice for our use case. The literature reveals a spectrum:

| Loss Type | Pros | Cons |
|-----------|------|------|
| **L1/L2 metric depth** | Provides absolute scale constraint | Requires metric depth; errors propagate directly |
| **Scale-shift-aligned L1** | Works with relative depth; per-image alignment | Scale/shift must be optimized; can drift |
| **Pearson correlation** | Scale-shift invariant | No absolute constraint |
| **Ordinal/ranking** | Robust to scale errors; captures structure | No metric grounding; weak constraint |
| **Combined** | Best of both worlds | More hyperparameters to tune |

**Recommendation for our use case**: Use **Metric3D v2** (which takes focal length as input) to predict metric depth, then apply both an L1 metric depth loss and a Pearson correlation / ranking loss. This leverages our known intrinsics for metric depth while maintaining robustness via the ordinal term.

---

## 4. Feed-Forward 3D from Images

These methods directly predict 3D structure from images in a single forward pass, without per-scene optimization. They represent a paradigm shift from optimization-based approaches.

### 4.1 DUSt3R (Wang, Leroy, et al., CVPR 2024)

- **Key contribution**: Predicts dense 3D point maps from image pairs in a single forward pass. Given two images, outputs per-pixel 3D coordinates in a shared reference frame. No camera calibration or pose information needed at inference.
- **Architecture**: Two ViT encoders (shared weights) + cross-attention decoder. Takes two images, produces two point maps and two confidence maps in the coordinate frame of image 1.
- **Multi-view extension**: For >2 images, runs all pairs and performs global alignment via a differentiable optimization that minimizes reprojection error across all pairs. Can recover camera intrinsics and extrinsics.
- **Output**: Dense per-pixel 3D point maps + confidence. Also implicitly provides relative poses and focal lengths.
- **Relevance**: Can be used as an alternative to SfM for generating initial 3D point clouds and recovering/verifying poses. Since we already have calibrated cameras and known poses, its main value would be as a dense 3D prediction for Gaussian splat initialization, or to verify/refine our calibration.

### 4.2 MASt3R (Leroy, Wang, et al., ECCV 2024)

- **Key contribution**: Extends DUSt3R with local feature matching capabilities. In addition to 3D point maps, MASt3R predicts dense local feature descriptors, enabling robust correspondence matching between image pairs. This improves the global alignment step and enables better multi-view reconstruction.
- **Architecture**: DUSt3R architecture + additional feature head producing per-pixel descriptors.
- **Relevance**: The feature matching capabilities could help with robust multi-camera registration, especially if our extrinsic calibration needs verification or refinement for specific frames.

### 4.3 PixelSplat (Charatan et al., CVPR 2024)

- **Key contribution**: Directly predicts 3D Gaussian splats from image pairs in a single forward pass. Given two calibrated input views, predicts per-pixel Gaussian parameters (position, covariance, opacity, color) that can be rendered from novel viewpoints.
- **Architecture**: Epipolar transformer -- cross-attention along epipolar lines between two views, producing per-pixel Gaussian distributions along the ray.
- **Input**: Two images + known intrinsics + known relative pose.
- **Limitations**: Trained on specific datasets (RealEstate10K, ACID); limited generalization to novel domains. Two-view input limits reconstruction completeness.

### 4.4 MVSplat (Chen et al., ECCV 2024)

- **Key contribution**: Extends the feed-forward Gaussian splat prediction paradigm to handle arbitrary numbers of input views efficiently. Uses a cost-volume-based approach (plane sweeping) rather than epipolar attention, which scales better to many views.
- **Input**: Multiple images + known intrinsics + known poses.
- **Relevance**: The multi-view cost volume approach is well-suited to our 12-camera setup. If fine-tuned on mining/outdoor data, could provide fast per-frame Gaussian predictions.

### 4.5 Splatt3R (Smart et al., arXiv 2024)

- **Key contribution**: Combines DUSt3R-style 3D prediction with Gaussian splatting output. Predicts 3D Gaussians directly from image pairs without requiring known camera parameters.
- **Architecture**: Built on MASt3R, adds Gaussian parameter prediction heads on top of the 3D point maps.

### 4.6 MVSGaussian (Liu et al., ECCV 2024)

- **Key contribution**: Combines multi-view stereo (MVS) cost volumes with Gaussian splatting for generalizable novel view synthesis. Uses a cascaded cost volume approach inspired by CasMVSNet to predict depth, then constructs Gaussians via unprojection.
- **Key insight**: The hybrid rendering approach handles the sparsity of initial Gaussians by using a lightweight neural renderer to fill in gaps, then distilling back to pure Gaussian representation.

### 4.7 LRM & InstantMesh (Hong et al., 2024; Xu et al., 2024)

- **LRM**: First large-scale feed-forward model for single-image 3D reconstruction. Uses a ViT encoder + transformer decoder to predict a NeRF (triplane representation) from a single image. Object-centric, not scene-level.
- **InstantMesh**: Extends LRM with multi-view input and FlexiCubes mesh extraction.

### 4.8 Assessment of Feed-Forward Methods for Our Use Case

Feed-forward methods are **not the primary recommendation** for our use case because:
1. They are mostly trained on indoor/object-centric data and may not generalize to mining environments.
2. We have strong camera calibration that per-scene optimization methods can fully exploit.
3. Per-scene optimization (3DGS) with depth priors can achieve higher quality than feed-forward methods for a specific scene.

However, feed-forward methods could serve as **initialization** for subsequent optimization, or for **rapid preview** generation.

---

## 5. Depth Unprojection Pipelines

The "classical" approach to incorporating monocular depth into 3D reconstruction follows a straightforward pipeline: predict depth -> unproject to 3D -> merge/fuse point clouds -> initialize 3DGS or extract mesh.

### 5.1 The Unprojection Process

Given an RGB image with known intrinsics K, known extrinsics [R|t], a depth map D, and distortion parameters:

1. **Undistort**: Apply inverse distortion model to pixel coordinates to get normalized image coordinates.
2. **Back-project**: For each pixel (u, v) with depth d: p_cam = d * K^{-1} * [u, v, 1]^T
3. **Transform to world**: p_world = R^{-1} * (p_cam - t)

For our rig with cam-to-vehicle transforms and GPS/IMU vehicle poses:
p_world = T_{world<-vehicle} * T_{vehicle<-cam} * p_cam

### 5.2 Distortion Handling

Our 8-parameter distortion model requires careful handling:
- Must **undistort pixel coordinates before unprojection**.
- **Recommendation**: Pre-undistort images before running monocular depth estimation. The depth model expects undistorted images (trained on pin-hole-like images). Feed undistorted images to depth model, then unproject using undistorted intrinsics.

### 5.3 Using Unprojected Depth for 3DGS Initialization

1. Predict monocular depth for each image (after undistortion).
2. Scale-shift align mono depth to metric scale (see S6).
3. Unproject each depth map to a colored point cloud using known K and [R|t].
4. Merge point clouds from all cameras/frames, optionally with voxel downsampling.
5. Initialize 3DGS from merged point cloud.
6. Optimize 3DGS with standard photometric loss + depth regularization losses.

### 5.4 TSDF Fusion

For mesh extraction, depth maps can be fused via Truncated Signed Distance Function (TSDF) integration. Libraries: Open3D (`ScalableTSDFVolume`), VDBFusion, NVIDIA nvblox.

**Limitation for mining/quarry**: TSDF requires choosing a voxel resolution and truncation distance. For large-scale outdoor scenes (hundreds of meters), memory can be prohibitive. Gaussian splatting may be more practical for the scene scale.

### 5.5 Poisson Surface Reconstruction

Alternative to TSDF: compute normals for the point cloud (from depth map gradients or from Metric3D v2 which predicts normals), then run Screened Poisson Reconstruction (Kazhdan & Hoppe, 2013). Produces watertight meshes but can over-smooth and struggles with open scenes.

---

## 6. The Scale Alignment Problem

This is arguably the **most critical practical challenge** when combining monocular depth with calibrated 3D reconstruction.

### 6.1 Problem Statement

Affine-invariant mono depth models predict depth up to an unknown per-image scale s_i and shift t_i:
d_metric = s_i * d_predicted + t_i

The scale and shift vary per image because the model normalizes output for numerical stability, different image content produces different scale/shift, and the model has no concept of metric units.

### 6.2 Per-Image Least-Squares Scale-Shift Fitting

Given sparse metric 3D points (from SfM, LiDAR, or known geometry), project them into each image to obtain pixel-wise metric depth samples, then solve for s_i, t_i via least squares.

**Sources of metric depth for alignment**:
- **Sparse SfM points**: Run COLMAP on the images; project triangulated 3D points into each image.
- **LiDAR**: If available on the rig. Optimal for alignment.
- **GPS-derived baselines**: Inter-camera baselines provide metric scale reference via stereo matching.
- **Known geometry**: In mining/quarry, known distances (bench heights, road widths, equipment dimensions).

### 6.3 Robust Fitting

- **RANSAC**: Fit scale-shift to random subsets, select inlier consensus.
- **Huber loss**: Replace L2 with Huber loss in the fitting.
- **Median-based**: Fit scale = median(metric/mono), shift via median of residuals.
- **Trimmed least squares**: Fit, remove worst 10-20% outliers, refit.

### 6.4 Joint Optimization

Rather than fixing scale/shift before 3DGS training, optimize them jointly:
- Include per-image s_i, t_i as learnable parameters during 3DGS optimization.
- **MonoSDF approach**: Analytically compute optimal s_i, t_i at each iteration via closed-form solution. Avoids making them learnable parameters.

### 6.5 Metric Depth Models That Avoid This Problem

| Approach | Scale Alignment Needed? | Requirements | Risk |
|----------|------------------------|--------------|------|
| Depth Anything v2 (relative) | **Yes**, per-image s_i, t_i | Sparse metric reference | Alignment errors propagate |
| Marigold (relative) | **Yes**, per-image s_i, t_i | Sparse metric reference | Same as above |
| ZoeDepth (metric) | **No**, but may be inaccurate | Nothing extra | Domain gap -> metric errors |
| Metric3D v2 (metric) | **No** | Focal length (known) | Domain gap -> metric errors |
| UniDepth (metric) | **No** | Nothing (or provide intrinsics) | Domain gap -> metric errors |

**Practical recommendation**: Run both Metric3D v2 (metric) and Depth Anything v2 (relative + alignment). Compare. If Metric3D v2 produces reasonable metric depth for the mining environment, use it directly.

### 6.6 Multi-View Consistency

Per-image scale-shift alignment does not guarantee multi-view consistency. Solutions:
- **Joint optimization** with 3DGS naturally enforces multi-view consistency through the shared Gaussian representation.
- **Cross-view alignment**: Check consistency in overlap regions and apply smoothing/averaging.
- **Global alignment**: Minimize cross-view reprojection errors jointly over all scale-shift parameters.

---

## 7. Recommendations for Our Use Case

### 7.1 System Configuration

- **12 calibrated cameras** with known intrinsics (fx, fy, cx, cy, 8-param distortion) and extrinsics (cam-to-vehicle transforms).
- **GPS/IMU** providing vehicle-to-world transforms per timestamp.
- **Environment**: Outdoor mining/quarry -- large scale, open terrain, challenging lighting, dust, textureless rock surfaces, steep terrain.
- **Goal**: 3D Gaussian splat reconstruction.

### 7.2 Recommended Pipeline

#### Phase 1: Data Preparation
1. **Undistort** all images using the 8-parameter distortion model.
2. **Compute camera-to-world transforms**: Compose T_{world<-vehicle} * T_{vehicle<-cam} for each camera at each timestamp.
3. **Convert to COLMAP format**: 3DGS expects COLMAP-style camera models. Include the undistorted intrinsics (PINHOLE model).
4. **Optionally run COLMAP SfM** on undistorted images to triangulate sparse 3D points (fix poses, only triangulate). Provides sparse metric 3D points for scale alignment.

#### Phase 2: Monocular Depth Estimation
1. **Primary: Depth Anything v2 (ViT-Large)** -- fast, robust, excellent zero-shot generalization.
2. **Secondary: Metric3D v2** -- provide focal length from calibration; get metric depth directly.
3. **Optional: Marigold** for uncertainty estimates on challenging scenes.

#### Phase 3: Scale-Shift Alignment (if using relative depth)
1. Project sparse COLMAP 3D points into each image.
2. Fit per-image scale-shift via robust least squares (RANSAC or Huber).
3. Alternatively, use GPS/IMU-derived baselines between cameras for metric reference.
4. Validate by comparing aligned mono depth with Metric3D v2 output.

#### Phase 4: 3DGS Initialization + Training
1. **Dense initialization**: Unproject aligned depth maps from all images/cameras into a merged point cloud. Voxel downsample (5-10 cm resolution for quarry scale).
2. **Initialize 3DGS** from dense point cloud.
3. **Training losses**:
   - **Photometric**: L1 + D-SSIM
   - **Hard depth loss**: L1 between rendered depth and aligned mono depth (or metric depth directly)
   - **Soft depth loss**: Pearson correlation (scale-shift invariant)
   - **Depth ranking loss** (optional): Ordinal consistency for pixel pairs
   - **Distortion loss**: Concentrate Gaussians along rays
   - **Normal consistency** (optional): If using Metric3D v2 normals
4. **FSGS-style** depth-guided Gaussian unprojection for adaptive densification.

#### Phase 5: Post-Processing
1. **Floater removal**: Prune Gaussians with low opacity or multi-view depth inconsistency.
2. **LOD/compression**: Hierarchical Gaussian representations for large scenes.

### 7.3 Key Considerations for Mining/Quarry Environment

1. **Scale**: Hundreds of meters. Consider spatial blocking for very large scenes.
2. **Textureless surfaces**: Rock faces, dirt roads -- mono depth priors are especially valuable here.
3. **Sky**: Mask sky pixels (segmentation model) and exclude from depth loss.
4. **Dust/haze**: Consider per-image appearance embeddings (NeRF-W style).
5. **Dynamic objects**: Mask moving vehicles/personnel via segmentation.
6. **GPS/IMU accuracy**: RTK ~2cm, standard ~2-5m. May need bundle adjustment refinement.
7. **Far-range depth**: Less reliable. Consider depth-dependent loss weighting: w(d) = 1/d.

### 7.4 Software Stack

| Component | Recommended Tool |
|-----------|-----------------|
| 3DGS Training | gsplat or original 3DGS |
| Depth Estimation | Depth Anything v2 |
| Metric Depth | Metric3D v2 |
| SfM | COLMAP |
| Point Cloud Processing | Open3D |
| Camera Undistortion | OpenCV |
| Sky Segmentation | SegFormer or SAM |

### 7.5 Ranking of Most Relevant Papers

For our specific use case (calibrated multi-camera rig, known poses, outdoor industrial, want Gaussian splats):

1. **3D Gaussian Splatting** (Kerbl et al., 2023) -- Base method
2. **DNGaussian** (Li et al., 2024) -- Depth-regularized 3DGS, dual hard/soft loss
3. **FSGS** (Zhu et al., 2024) -- Depth-guided initialization and densification
4. **Depth Anything v2** (Yang et al., 2024) -- Best relative depth backbone
5. **Metric3D v2** (Hu et al., 2024) -- Metric depth with known intrinsics
6. **MonoSDF** (Yu et al., 2022) -- Scale-shift invariant depth loss formulation
7. **SparseGS** (Xiong et al., 2023) -- Floater pruning with depth
8. **DUSt3R** (Wang et al., 2024) -- Alternative dense 3D initialization
9. **Marigold** (Ke et al., 2024) -- Depth with uncertainty estimates

---

## References

1. Azinovic, D., Martin-Brualla, R., Goldman, D. B., Niessner, M., & Thies, J. (2022). Neural RGB-D Surface Reconstruction. *3DV*.
2. Bhat, S. F., Birkl, R., Wofk, D., Wonka, P., & Muller, M. (2023). ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth. *arXiv:2302.12288*.
3. Charatan, D., Li, S., Tagliasacchi, A., & Sitzmann, V. (2024). PixelSplat: 3D Gaussian Splats from Image Pairs for Scalable Generalizable 3D Reconstruction. *CVPR*.
4. Chen, Y., Xu, H., Zheng, C., Zhuang, B., Pollefeys, M., Geiger, A., Cai, T., & Reid, I. (2024). MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images. *ECCV*.
5. Chung, J., Oh, J., & Lee, K. M. (2024). Depth-Regularized Optimization for 3D Gaussian Splatting in Few-Shot Images. *CVPR Workshop*.
6. Deng, K., Liu, A., Zhu, J.-Y., & Ramanan, D. (2022). Depth-supervised NeRF: Fewer Views and Faster Training for Free. *CVPR*.
7. Hong, Y., Zhang, K., Gu, J., Bi, S., Zhou, Y., Liu, D., Liu, F., Sunkavalli, K., Bui, T., & Tan, H. (2024). LRM: Large Reconstruction Model for Single Image to 3D. *ICLR*.
8. Hu, M., Yin, W., Zhang, L., Ren, B., Cai, D., & He, X. (2024). Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-Shot Metric Depth and Surface Normal Estimation. *arXiv:2404.15506*.
9. Ke, B., Obukhov, A., Huang, S., Mez, N., Dauber, M., & Schindler, K. (2024). Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation. *CVPR*.
10. Kerbl, B., Kopanas, G., Leimkuhler, T., & Drettakis, G. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering. *SIGGRAPH*.
11. Leroy, V., Wang, Y., Revaud, J., et al. (2024). Grounding Image Matching in 3D with MASt3R. *ECCV*.
12. Li, J., Li, B., & Lu, Y. (2024). DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Global-Local Depth Normalization. *CVPR*.
13. Liu, T., et al. (2024). MVSGaussian: Fast Generalizable Gaussian Splatting Reconstruction from Multi-View Stereo. *ECCV*.
14. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. *ECCV*.
15. Niemeyer, M., Barron, J. T., Mildenhall, B., Sajjadi, M. S. M., Geiger, A., & Radwan, N. (2022). RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Few Input Views. *CVPR*.
16. Piccinelli, L., Yang, Y., Sakaridis, C., Segu, M., Li, S., Van Gool, L., & Yu, F. (2024). UniDepth: Universal Monocular Metric Depth Estimation. *CVPR*.
17. Ranftl, R., Bochkovskiy, A., & Koltun, V. (2021). Vision Transformers for Dense Prediction. *ICCV*.
18. Ranftl, R., Lasinger, K., Hafner, D., Schindler, K., & Koltun, V. (2020). Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer. *TPAMI*.
19. Smart, B., et al. (2024). Splatt3R: Zero-shot Gaussian Splatting from Uncalibrated Image Pairs. *arXiv*.
20. Wang, S., Leroy, V., Cabon, Y., Weinzaepfel, P., & Revaud, J. (2024). DUSt3R: Geometric 3D Vision Made Easy. *CVPR*.
21. Wynn, J. & Turmukhambetov, D. (2023). DiffusioNeRF: Regularizing Neural Radiance Fields with Denoising Diffusion Models. *CVPR*.
22. Xiong, H., et al. (2023). SparseGS: Real-Time 360 Sparse View Synthesis using Gaussian Splatting. *arXiv*.
23. Xu, J., et al. (2024). InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models. *arXiv*.
24. Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., & Zhao, H. (2024). Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data. *CVPR*.
25. Yang, L., Kang, B., Huang, Z., Zhao, Z., Xu, X., Feng, J., & Zhao, H. (2024). Depth Anything V2. *NeurIPS*.
26. Yin, W., Zhang, J., Wang, O., Niklaus, S., Mai, L., Chen, S., & Shen, C. (2023). Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image. *ICCV*.
27. Yu, Z., Peng, S., Niemeyer, M., Sattler, T., & Geiger, A. (2022). MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction. *NeurIPS*.
28. Zhu, Z., Fan, Z., Jiang, Y., & Wang, Z. (2024). FSGS: Real-Time Few-Shot View Synthesis using Gaussian Splatting. *CVPR*.
