# Spatia: Video Generation with Updatable Spatial Memory

**Jinjing Zhao¹\*** · **Fangyun Wei²\*†** · **Zhening Liu³** · **Hongyang Zhang⁴** · **Chang Xu¹†** · **Yan Lu²**

¹The University of Sydney · ²Microsoft Research · ³HKUST · ⁴University of Waterloo

🔗 https://zhaojingjing713.github.io/Spatia/

> \*Equal contribution. †Corresponding author.

---

## Abstract

Existing video generation models struggle to maintain long-term spatial and temporal consistency due to the dense, high-dimensional nature of video signals. To overcome this limitation, we propose **Spatia**, a spatial memory–aware video generation framework that explicitly preserves a 3D scene point cloud as persistent spatial memory. Spatia iteratively generates video clips conditioned on this spatial memory and continuously updates it through visual SLAM. This dynamic–static disentanglement design enhances spatial consistency throughout the generation process while preserving the model's ability to produce realistic dynamic entities. Furthermore, Spatia enables applications such as explicit camera control and 3D-aware interactive editing, providing a geometrically grounded framework for scalable, memory-driven video generation.

---

## 1. Introduction

Video generation has emerged as a foundational technique powering a wide spectrum of tasks. Recent advances in video generation foundation models have significantly improved the quality and controllability of short-duration video synthesis. On the other hand, there is a growing need to extend these models toward long-horizon video generation, enabling applications that require temporal consistency and persistent memory, such as world models, AI-driven game generation, and embodied AI.

Unlike LLMs, video generation models encounter intrinsic difficulties in encoding long-term historical information, primarily due to the dense and high-dimensional nature of video signals. For instance, a short 5-second 480P (640 × 480) video at 24 FPS—consisting of 120 frames—already corresponds to 40 × 30 × 30 = **36,000 spatio-temporal tokens** when using a video encoder with a spatial downsampling factor of 16 and a temporal downsampling factor of 4.

> By comparison, 36,000 tokens can represent around 27,000 words. With the same number of tokens, a video generation model can capture only about 5 seconds of visual history, whereas an LLM can encompass a context equivalent to 27,000 words.

In this work, we introduce an explicit memory mechanism designed to achieve consistent and long-horizon video generation. The process begins by estimating an initial 3D scene point cloud from the conditional input image, which serves as the **spatial memory** of the scene. We then iteratively perform two key steps:

1. Generate a new video clip conditioned on both the current 3D scene point cloud and the previously generated video clip, ensuring temporal and spatial consistency across iterations.
2. Update the scene point cloud using visual SLAM algorithms based on both newly generated and previously generated frames, thereby incorporating new content while preserving existing scene information.

We name our approach **Spatia**, short for *spatial memory–aware video generation*. Spatia enjoys the following key characteristics:

- **(a) Dynamic–Static Disentanglement.** Spatia preserves a scene point cloud as spatial memory while simultaneously generating dynamic entities that interact coherently with the scene.
- **(b) Spatially Consistent Generation.** By retrieving spatial memory, Spatia can generate diverse video sequences depicting the same location from different viewpoints while preserving a consistent spatial structure.
- **(c) Explicit Camera Control.** Spatia achieves camera control in an explicit and geometrically grounded manner by directly applying the desired camera path to the 3D scene point cloud and rendering a corresponding 2D point cloud sequence.
- **(d) 3D-Aware Interactive Editing.** Users can interactively edit the scene before generation—for example, by removing or modifying specific objects—and such edits are directly reflected in the generated videos.

---

## 2. Related Works

### Video Generation Models

The field of video generation has evolved rapidly, progressing from early UNet-based latent diffusion models to large-scale Diffusion Transformers. While bidirectional models employing global spatio-temporal attention achieve impressive fidelity, their quadratic computational complexity fundamentally limits them to short-clip generation. To generate arbitrarily long sequences, autoregressive frameworks have been proposed, which iteratively synthesize new content conditioned on previously generated frames.

### Camera Control in Video Generation

Precise camera control has become a key goal in video synthesis. One line of work conditions generation on explicit camera parameters—for example, AnimateDiff employs motion LoRAs to learn specific camera trajectories. For finer-grained control, geometry-aware approaches use 3D information—such as rendered point clouds—to provide dense spatial guidance for camera path generation.

### Long-term Memory Modeling

A central strategy for improving the long-term memory capacity of LLMs lies in expanding their native context window. In video generation, the bidirectional spatiotemporal attention used in most diffusion models prevents standard KV caching, thereby severely limiting the context window. Recent works have introduced memory-based architectures to preserve long-term spatial consistency.

### Scene Point Cloud Estimation

Recent progress in visual geometry estimation is led by Dust3R, which unifies pairwise pose and geometry estimation but encounters a costly O(N²) global alignment bottleneck. This motivates follow-up works to develop more scalable solutions. In parallel, universal end-to-end models eliminate the pairwise dependency, employing large Transformers to infer globally consistent 3D geometry and camera parameters.

---

## 3. Method

### Problem Formulation

The objective of Spatia is to endow a video generation model with persistent spatial memory, enabling it to produce videos that are both spatially and temporally consistent. Spatia formulates the entire framework as a **multi-modal conditional generation problem**, operating in two stages:

1. Generating a video clip conditioned on multi-modal inputs—including text instructions, geographically retrieved information from the spatial memory, and either an initial image or previously generated clips.
2. Updating the spatial memory to incorporate newly generated content, ensuring that subsequent generations remain geometrically consistent with the evolving scene.

### 3.1. Training

#### Training Data

For a given training video **V**, we decompose it into three parts:

$$\mathcal{V} = \{T\}^N \cup \{P\}^M \cup \{C\}^O$$

where:
- $\{T\}^N$ — **Target frames**: the clip to be generated by the model
- $\{P\}^M$ — **Preceding frames**: the clip immediately before the target, providing temporal context
- $\{C\}^O$ — **Candidate frames**: remaining frames, serving as potential references for spatial and geometric consistency

#### 3.1.1. View-Specific Scene Point Cloud Estimation

**Scene Point Cloud Estimation.** A frame is randomly sampled from the candidate-frame set and **MapAnything** is used to estimate a scene point cloud $\mathcal{S}$. If the training video contains dynamic entities, a segmentation process is performed to remove them before point cloud estimation:

1. **Keye-VL-1.5** identifies dynamic entities and generates corresponding text prompts.
2. **ReferDINO** segments out these dynamic entities.

**Per-Frame Camera Pose Estimation.** The camera pose for each frame in $\{T\}^N \cup \{P\}^M \cup \{C\}^O$ is estimated using MapAnything, denoted as $\{\theta_T\}^N$, $\{\theta_P\}^M$, and $\{\theta_C\}^O$.

**View-Specific Scene Point Clouds.** Given $\mathcal{S}$ and the per-frame camera poses, each camera pose is applied to $\mathcal{S}$ to render the scene from the corresponding viewpoint, yielding view-specific scene point clouds $\{\mathcal{S}_T\}^N$, $\{\mathcal{S}_P\}^M$, and $\{\mathcal{S}_C\}^O$.

#### 3.1.2. Reference Frame Retrieval

The objective is to select up to **K** of the most spatially relevant frames from $\{C\}^O$ as reference frames for the target clip $\{T\}^N$. Spatial correspondence is computed using their associated scene point clouds via **3D IoU**:

$$\text{SPATIALOVERLAP}(x, y): \quad y' \leftarrow \text{Register}(y, x), \quad s \leftarrow \text{3DIoU}(x, y')$$

A frame $C_j$ is selected as a reference if the spatial overlap score $s(T_i, C_j) > \epsilon$.

#### 3.1.3. Architecture

Spatia adopts a **multi-modal conditional generation framework** to generate target clip $\{T\}^N$ conditioned on:
- Preceding video clip $\{P\}^M$
- Scene point clouds $\{\mathcal{S}_T\}^N$ and $\{\mathcal{S}_P\}^M$
- Retrieved reference frames $\{R\}^K$
- Text instruction $\mathcal{T}$

**Token Extraction:**
- Video inputs $\{T\}^N$ and $\{P\}^M$ → **Wan2.2** video encoder → spatio-temporal tokens $X_T$, $X_P$
- Reference frames $\{R\}^K$ → same video encoder → token sequence $X_R$
- Scene point clouds → projected onto 2D image plane → video encoder → $X_{\mathcal{S}_T}$, $X_{\mathcal{S}_P}$
- Text instruction $\mathcal{T}$ → text encoder → text tokens $X_\mathcal{T}$

**Network Structure.** Spatia includes **8 network blocks**, each containing one **ControlNet** block operating in parallel with **four main blocks**. Each main block consists of:
- Self-attention layer
- Cross-attention layer (text tokens as keys/values)
- FFN

Each ControlNet block adopts the same architecture but appends a **projector (MLP)** after the FFN.

**Training Objective (Flow Matching):**

$$\mathcal{L} = \mathbb{E}_{t, x_0, X_T} \|v_t - u_t\|^2$$

where $u_t = dx_t/dt$ is the ground-truth velocity and $v_t$ is the predicted velocity.

### 3.2. Inference

Spatia enables iterative user interaction. At each iteration:

1. The user specifies a **text instruction** and a **camera trajectory** based on the current 3D scene point cloud.
2. A projection video is rendered along the desired trajectory, conditioning the generation.
3. The newly generated content is used to **update the spatial memory** via MapAnything.

---

## 4. Experiments

### Implementation Details

| Component | Details |
|---|---|
| Backbone | Wan2.2, 5B parameters |
| Training data | RealEstate10K (40K videos) + SpatialVID HD (10K videos), 720P |
| Stage 1 | Train ControlNet blocks, 8,000 iterations, lr = 1e-5, main network frozen |
| Stage 2 | Fine-tune main blocks with LoRA (rank=64), 5,000 iterations, lr = 1e-4 |
| Batch size | 64 on 64× AMD MI250 GPUs |
| Output length | 81 frames (1st iteration) / 72 frames (subsequent iterations, conditioned on 9 prev. frames) |

### 4.1. Main Results

#### Visual Quality — WorldScore Benchmark

| Method | Avg Score | Static Score | Dynamic Score | Camera Ctrl |
|---|---|---|---|---|
| **Static scene generation models** | | | | |
| WonderJourney | 54.19 | 63.75 | 44.63 | 84.60 |
| InvisibleStitch | 51.95 | 61.12 | 42.78 | 93.20 |
| WonderWorld | 61.79 | 72.69 | 50.88 | 92.98 |
| Voyager | 66.08 | 77.62 | 54.53 | 85.95 |
| **Foundation video generation models** | | | | |
| VideoCrafter2 | 50.03 | 52.57 | 47.49 | 28.92 |
| EasyAnimate | 52.25 | 52.85 | 51.65 | 26.72 |
| Allegro | 53.64 | 55.31 | 51.97 | 24.84 |
| CogVideoX-I2V | 60.64 | 62.15 | 59.12 | 38.27 |
| Vchitect-2.0 | 40.38 | 42.28 | 38.47 | 26.55 |
| LTX-Video | 55.99 | 55.44 | 56.54 | 25.06 |
| Wan2.1 | 55.21 | 57.56 | 52.85 | 23.53 |
| **Spatia (Ours)** | **69.73** | **72.63** | **66.82** | **75.66** |

#### Visual Quality — RealEstate Test Set (Table 2)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| SEVA | 13.07 | 0.515 | 0.445 |
| VMem | 14.62 | 0.522 | 0.426 |
| ViewCrafter | 15.78 | 0.580 | 0.396 |
| FlexWorld | 16.25 | 0.593 | 0.370 |
| Voyager | 17.79 | 0.636 | 0.297 |
| **Spatia (Ours)** | **18.58** | **0.646** | **0.254** |

#### Memory Mechanism Evaluation — Closed-Loop Generation (Table 3)

| Method | PSNR_C ↑ | SSIM_C ↑ | LPIPS_C ↓ | Match Acc ↑ |
|---|---|---|---|---|
| ViewCrafter | 14.79 | 0.481 | 0.365 | 0.447 |
| FlexWorld | 12.20 | 0.428 | 0.598 | 0.377 |
| Voyager | 17.66 | 0.540 | 0.380 | 0.507 |
| **Spatia (Ours)** | **19.38** | **0.579** | **0.213** | **0.698** |

### 4.2. Ablation Studies

#### Impact of Scene Video and Reference Frames (Table 4)

| Scene Video | Reference Frames | Camera Control | PSNR_C | SSIM_C | LPIPS_C |
|---|---|---|---|---|---|
| ✗ | ✗ | 58.81 | 15.55 | 0.444 | 0.379 |
| ✓ | ✗ | 80.13 | 17.18 | 0.500 | 0.295 |
| ✗ | ✓ | 61.38 | 15.64 | 0.444 | 0.393 |
| ✓ | ✓ | **84.47** | **19.38** | **0.579** | **0.213** |

#### Number of Reference Frames (Table 5)

| # Reference Frames | PSNR_C | SSIM_C | LPIPS_C | Match Acc |
|---|---|---|---|---|
| 1 | 17.50 | 0.537 | 0.284 | 0.592 |
| 3 | 17.85 | 0.540 | 0.275 | 0.606 |
| 5 | 18.48 | 0.556 | 0.248 | 0.640 |
| **7** | **19.38** | **0.579** | **0.213** | **0.698** |

#### Long-Horizon Generation (Table 6)

| Method | #Clips | Camera Control | PSNR_C | SSIM_C | LPIPS_C |
|---|---|---|---|---|---|
| Wan2.2 | 2 | 56.87 | 13.00 | 0.377 | 0.521 |
| Wan2.2 | 4 | 46.43 | 11.32 | 0.328 | 0.611 |
| Wan2.2 | 6 | 49.97 | 10.74 | 0.310 | 0.644 |
| **Spatia** | **2** | **84.47** | **19.38** | **0.579** | **0.213** |
| **Spatia** | **4** | **83.97** | **18.23** | **0.546** | **0.253** |
| **Spatia** | **6** | **83.41** | **18.04** | **0.541** | **0.259** |

#### Point Cloud Density (Table 7)

| Cube Side Length (m) | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| 0.01 | 18.58 | 0.646 | 0.254 |
| 0.03 | 17.10 | 0.614 | 0.313 |
| 0.05 | 16.35 | 0.596 | 0.349 |
| 0.07 | 15.97 | 0.585 | 0.370 |

---

## 5. Conclusion

We introduce **Spatia**, a spatial memory–aware video generation framework that enables consistent, long-horizon synthesis. By maintaining an explicit 3D scene point cloud as persistent memory and iteratively updating it during generation, Spatia captures long-term geometric structure that conventional video models cannot preserve. This memory mechanism ensures spatial consistency across revisited locations, supports coherent dynamic content, and enables explicit camera control through 3D-aware conditioning. Extensive experiments demonstrate that Spatia significantly enhances long-horizon consistency while maintaining high visual quality in the generated videos.

---

## Supplementary Material

### 6. More Implementation Details

#### Reference Frame Retrieval Algorithm

```
Algorithm 1: Reference Frame Retrieval

Input:  Target frames {T}^N, candidate frames {C}^O,
        view-specific scene point clouds {S_T}^N and {S_C}^O,
        threshold ε, maximum number of reference frames K
Output: Retrieved reference-frame set {R}

Initialize {R} ← ∅
for each target frame T_i ∈ {T}^N do
    if i mod K ≠ 0 then
        break  ▷ Operate every K frames.
    end if
    Initialize s ← 0        ▷ Maximal spatial overlap score.
    Initialize R̂ ← ∅       ▷ Empty reference frame.
    Identify the scene map S_{T_i} ∈ {S_T}^N
    for each candidate frame C_j ∈ {C}^O do
        Identify the scene map S_{C_j} ∈ {S_C}^O
        s(T_i, C_j) ← SPATIALOVERLAP(S_{T_i}, S_{C_j})
        if s(T_i, C_j) > s then
            s ← s(T_i, C_j)
            R̂ ← C_j
        end if
    end for
    if s > ε then
        {R} ← {R} ∪ R̂
    end if
end for
return {R}

function SPATIALOVERLAP(x, y):
    y' ← Register(y, x)   ▷ Register y to x space.
    s  ← 3DIoU(x, y')
    return s
```

#### Augmentation of Preceding-Frame Latents

To alleviate the distribution gap between training (ground-truth preceding frames) and inference (model-generated frames), a noise augmentation strategy is applied: a timestep $t_{aug} \in [0, 50]$ is sampled from a low-noise interval and the corresponding noise is added to the clean preceding-frame latents.

#### Match Accuracy

Match Accuracy quantifies the structural and spatial correspondence between two frames using **RoMa** (a robust dense feature-matching algorithm). After obtaining the correspondence map between $I_{first}$ and $I_{last}$, low-confidence matches are discarded and the final match accuracy is normalized by the number of high-confidence self-matches of $I_{first}$.

#### Dynamic-Static Disentanglement at Inference

During inference, **SAM2** is applied to track and segment dynamic entities. The resulting segmentation masks are used to exclude dynamic regions when updating the spatial memory via MapAnything.

---

## References

| # | Citation |
|---|---|
| [22] | Duan et al. *WorldScore: A Unified Evaluation Benchmark for World Generation.* arXiv:2504.00983, 2025. |
| [40] | Huang et al. *Voyager: Long-range and World-consistent Video Diffusion for Explorable 3D Scene Generation.* arXiv:2506.04225, 2025. |
| [42] | Keetha et al. *MapAnything: Universal Feed-forward Metric 3D Reconstruction.* arXiv:2509.13414, 2025. |
| [43] | Kerbl et al. *3D Gaussian Splatting for Real-Time Radiance Field Rendering.* ACM Trans. Graph., 2023. |
| [51] | Li et al. *VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory.* arXiv:2506.18903, 2025. |
| [54] | Liang et al. *ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations.* ICCV, 2025. |
| [56] | Lipman et al. *Flow Matching for Generative Modeling.* arXiv:2210.02747, 2022. |
| [74] | Ravi et al. *SAM 2: Segment Anything in Images and Videos.* arXiv:2408.00714, 2024. |
| [87] | Team Wan et al. *Wan: Open and Advanced Large-Scale Video Generative Models.* arXiv:2503.20314, 2025. |
| [90] | Wang et al. *SpatialVID: A Large-Scale Video Dataset with Spatial Annotations.* arXiv:2509.09676, 2025. |
| [92] | Wang et al. *DUSt3R: Geometric 3D Vision Made Easy.* CVPR, 2024. |
| [101] | Yang et al. *Kwai Keye-VL 1.5 Technical Report.* arXiv:2509.01563, 2025. |
| [113] | Yu et al. *ViewCrafter: Taming Video Diffusion Models for High-Fidelity Novel View Synthesis.* arXiv:2409.02048, 2024. |
| [115] | Zhang et al. *Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet).* 2023. |
| [122] | Zhou et al. *Stereo Magnification: Learning View Synthesis Using Multiplane Images.* arXiv:1805.09817, 2018. |

python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version (torch built with):', torch.version.cuda); print('Device count:', torch.cuda.device_count())"