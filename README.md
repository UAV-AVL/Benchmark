<div align="center">
  <video src="https://github.com/user-attachments/assets/396b94f9-3f8c-43fe-a358-17a73b812e14" controls="controls" width="600" height="360"></video>
  <!--<img src="overview.png" alt="logo" width="400" height="auto" />-->
  <h1>The First Large-scale Benchmark for UAV Visual Localization under Low-altitude Multi-view Observation Conditions</h1>
  <p>
    AnyVisLoc is a benchmark for UAV visual localization under low-altitude, multi-view observation conditions. It uses 2.5D aerial and satellite reference maps, and its baseline follows a unified image retrieval → image matching → Perspective-n-Point (PnP) localization framework.
  </p>
<p>
  🎉🎉🎉 <strong>News:</strong> Our paper has been accepted to <strong>CVPR 2026 Findings</strong>! 🎉🎉🎉
  <strong>The complete AnyVisLoc dataset is now publicly available, and the testing code has been upgraded.</strong>
  Thank you for your attention and support!
</p>
<p>
  If you find our work useful, please consider giving us a ⭐️. Your support means a lot to us! 🥰🥰🥰
</p>
<h4>
  <a href="https://openaccess.thecvf.com/content/CVPR2026F/papers/Ye_Exploring_the_best_way_for_UAV_visual_localization_under_Low-altitude_CVPRF_2026_paper.pdf">
    View Paper
  </a> 
  <span> · </span>
  <a href="https://openaccess.thecvf.com/content/CVPR2026F/supplemental/Ye_Exploring_the_best_CVPRF_2026_supplemental.pdf">
    Supplementary Material
  </a> 
  <span> · </span>
  <a href="https://pan.baidu.com/s/14vSQDydkmiTj2U1JpQH-Hg?pwd=fcw8">
    Download Dataset (Baidu NetDisk)
  </a>
  <a href="https://1drv.ms/f/c/fef40436afe2e5bb/IgC0Zat90dsgTY-2lNlDliGXAfuTZOYEeKcllr7VPD_F3eY?e=LJ5fAD">
    (Onedrive)
  </a>
  <span> · </span>

</h4>

</div>

<br />

<!-- Table of Contents -->

# 📒 Table of Contents
- [The AnyVisLoc Dataset](#about-the-dataset)
  * [UAV Images Examples](#UAV-Images)
  * [Reference Map Examples](#Reference-Maps)
  * [Dataset Features](#Dataset-Features)
  * [Ground Truth Preparation](#Ground-Truth-Preparation)
  * [Dataset Statistics](#Dataset-Statistics)
- [The Baseline Demo](#baseline)
  * [Supported Baseline Components](#Supported-Baseline-Components)
  * [Installation](#Installation)
  * [Download Files](#Download)
  * [Dataset Format](#Dataset-Format)
  * [Run the Full Pipeline](#Run-the-Full-Pipeline)
  * [Run Retrieval Only](#Run-Retrieval-Only)
  * [Run Matching and Localization Only](#Run-Matching-and-Localization-Only)
  * [Important Arguments](#Important-Arguments)
  * [Outputs](#Outputs)
  * [Test Your Own Dataset](#test_dataset)
  * [Test Your Visual Localization Approaches](#test_approaches)
- [FAQ](#FAQ)
- [Citation](#Citation)
- [License](#License)
- [Acknowledgments](#Acknowledgments)
- [Contact](#Contact)

<!-- About the AnyVisLoc Dataset -->
<a name="about-the-dataset"></a>
## 📸 AnyVisLoc Dataset

<!-- UAV Images Examples -->
<a name="UAV-Images"></a>
### ✈️ UAV Images

<div align="center">
  <img src="assets/overview_supp.png" alt="UAV Image Examples" />
</div>

<!-- Reference Map -->
<a name="Reference-Maps"></a>
### 🗺️ Reference Maps

<div align="center">
  <img src="assets/reference_map_new1.png" alt="Reference Map Examples" />
</div>

<!-- Dataset Features -->
<a name="Dataset-Features"></a>
### 🌟 Dataset Features
- **Large scale:** **20,077** full-resolution DJI UAV images from **24** scenes across China. The reference maps cover distinct regions ranging in coverage area from **10,000 $m^2$ to 9,000,000 $m^2$**.
- **Multi-altitude:** The dataset contains diverse UAV flight heights, ranging approximately from **6 m to 500 m**.
- **Multi-view:** The dataset covers common UAV imaging pitch angles from approximately **5° to 90°**, including both nadir and oblique views.
- **Multi-scene:** The dataset includes dense **urban** areas, towns and villages, typical **landmark** scenes, campuses, parks, **natural** scenes such as grasslands, farmland, and mountains, as well as mixed environments.
- **Multi-reference map:** The dataset provides two complementary types of **2.5D** reference maps. The **aerial map** provides high spatial resolution for high-precision localization, while the **satellite map** is a more broadly available reference source that does not require scene-specific aerial acquisition or reconstruction.
- **Multi-drone type:** DJI Mavic 2, Mavic 3, Mavic 3 Pro, Phantom 3, Phantom 4, Phantom 4 RTK, and Mini 4 Pro.
- **Others:** multiple weather conditions (☀️⛅☁️🌫️🌧️), seasons (🌻🍀🍂⛄), and illumination conditions (🌇🌆).

> **Note:** To further improve the diversity of the public release, we removed three scenes with limited geographic coverage and relatively single-view observations, and added two new scenes. The current release contains **24 scenes** and **20,077 UAV images**.

<a name="Ground-Truth-Preparation"></a>

### 🧭 Ground Truth Preparation

The ground-truth generation protocol is provided in [GROUND_TRUTH.md](docs/GROUND_TRUTH.md).

<a name="Dataset-Statistics"></a>

### 📊 Dataset Statistics

Detailed scene-level statistics are provided in [DATASET_STATISTICS.md](docs/DATASET_STATISTICS.md).

<!-- Running the baseline demo -->
<a name="baseline"></a>
## 🚩 The Baseline Demo

This repository provides a unified AnyVisLoc testing pipeline for UAV visual localization. The testing code provides three evaluation modes:

- **Full pipeline:** image retrieval → pixel matching → Perspective-n-Point (PnP) localization.
- **Retrieval only:** evaluates image-level retrieval without matching or PnP.
- **Matching + localization only:** evaluates pixel matching and Perspective-n-Point (PnP) localization using the ground-truth reference crop, without image retrieval.

### 🧰 Supported Baseline Components

The current release supports the following components:

| Component | Supported options |
| --- | --- |
| Image retrieval | [`CAMP`](https://github.com/Mabel0403/CAMP) |
| Pixel matching | [`RoMa`](https://github.com/Parskatt/RoMa), [`SP_LG`](https://github.com/cvg/LightGlue), [`SP_LG_GIM`](https://github.com/xuelunshen/gim), [`SP_LG_MINIMA`](https://github.com/LSXI7/MINIMA), [`ALIKED_LG`](https://github.com/Shiaoming/ALIKED), and [`DISK_LG`](https://github.com/cvlab-epfl/disk) |
| PnP solver | [`OpenCV`](https://github.com/opencv/opencv) implementations: `P3P` by default; `AP3P`, `EPNP` are also available through `--PnP_method` |
| Pose-prior setting | `yp` or `p` through `--pose_priori` |

> **Pose-prior protocol.** In practical UAV applications, coarse altitude and attitude measurements are often available from the onboard inertial navigation system. Under low-altitude oblique views, a usable altitude and pitch angle estimate is important for estimating the projected footprint and cropping an appropriate aerial or satellite reference region. Large uncompensated yaw differences can also substantially reduce retrieval and pixel-matching accuracy.
>
> `yp` is the easier and default setting: it uses coarse pitch, yaw, and altitude to estimate the view center, and yaw-aligns the reference map before retrieval and matching. `p` is the more challenging setting: it retains the pose-aware view-center estimate but does **not** yaw-rotate the reference map, leaving the yaw discrepancy to the retrieval and matching stages. `unknown` uses no pose prior. To study less reliable onboard measurements, users may perturb the released pose metadata before evaluation. These standardized settings are provided to make evaluations on AnyVisLoc more comparable; they do not prevent users from testing other prior assumptions or noise levels.

<!-- Installation -->
<a name="Installation"></a>
### ⚙️ Installation

Clone the project:

```bash
git clone https://github.com/UAV-AVL/Benchmark.git
cd Benchmark
```

We recommend creating a clean conda environment. The demo has been tested with **Python 3.10**, **PyTorch 2.5.1**, **CUDA 12.1**, and an NVIDIA RTX 4090 GPU.

```bash
conda create -n anyvisloc_demo python=3.10 -y
conda activate anyvisloc_demo
```

Install GDAL from conda-forge:

```bash
conda install -c conda-forge gdal
```

Install PyTorch first according to your CUDA version. For example, for CUDA 12.1:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If you use a different CUDA version, install the corresponding PyTorch build from the official PyTorch installation page before running `pip install -r requirements.txt`.

Optional acceleration packages such as `flash-attn` and `xformers` are not included in the basic `requirements.txt`, because they are highly dependent on the CUDA, PyTorch, GCC, and system environment. Install them only when your environment is compatible.

<!-- Download-->
<a name="Download"></a>

### ⬇️ Download Files

#### 1. Dataset

The complete AnyVisLoc dataset is publicly available through the official download portal. By downloading or using the dataset, you agree to the [AnyVisLoc Dataset License](docs/LICENSE_DATASET.md). Please place the downloaded files under a local dataset root, for example:

```text
./Data/AnyVisLoc/
```

#### 2. Model Weights

The default baseline configuration uses **CAMP** for image retrieval and supports **RoMa** and selectable sparse matchers for pixel-level matching. Required checkpoints can be downloaded from the original projects: [CAMP](https://drive.google.com/file/d/1qHjXr3VVQuJZ5kE5u7YrUB8id90Nv2GJ/view?usp=sharing), [RoMa](https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth), [DINOv2 ViT-L/14 (used by RoMa)](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth), [GIM LightGlue](https://github.com/xuelunshen/gim/raw/refs/heads/main/weights/gim_lightglue_100h.ckpt), and [MINIMA LightGlue](https://github.com/LSXI7/storage/releases/download/MINIMA/minima_lightglue.pth). Alternatively, all required checkpoints can be downloaded from our [Baidu NetDisk link](https://pan.baidu.com/s/15h9puhU7Xvokp7cuRf50ZA) (extraction code: `zpy9`).

Please keep the original filenames and place the checkpoints in the expected folders:

```text
Benchmark/
├── Retrieval_Models/
│   └── CAMP/
│       └── weights/
│           └── <CAMP checkpoint>.pth
└── Matching_Models/
    ├── RoMa/
    │   └── ckpt/
    │       ├── roma_outdoor.pth
    │       └── dinov2_vitl14_pretrain.pth
    └── Sparse_matchers/
        └── weights/
            ├── gim_lightglue_100h.ckpt
            └── minima_lightglue.pth
```

> **Note:** All third-party checkpoints remain subject to the licenses, terms of use, and citation requirements of their original authors.

<!-- Dataset Format -->
<a name="Dataset-Format"></a>

### 🧩 Dataset Format

The released AnyVisLoc dataset uses a scene-wise NPZ structure. Each scene folder contains one lightweight reference JSON file, two reference maps, two DSM files, and multiple UAV image samples.

```text
Data/AnyVisLoc/
├── Scene_01/
│   ├── L01_reference.json
│   ├── aerial_map.png
│   ├── aerial_dsm.npy
│   ├── satellite_map.png
│   ├── satellite_dsm.npy
│   ├── L01_0001.npz
│   ├── L01_0002.npz
│   ├── ...
├── Scene_02/
│   ├── L02_reference.json
│   ├── aerial_map.png
│   ├── aerial_dsm.npy
│   ├── satellite_map.png
│   ├── satellite_dsm.npy
│   ├── L02_0001.npz
│   ├── ...
└── ...
```

#### Reference files

Each `Lxx_reference.json` is **required by the current runners**. It is used to discover scene folders and, for each sample, resolve the reference data selected by `--reference_mode` before the map and DSM are loaded lazily.

The current testing code supports:

- `aerial`: high-resolution aerial orthophoto + aerial DSM.
- `satellite`: satellite image + satellite DSM.

For each reference mode, the JSON file records the relative paths to the reference map and DSM, the map and DSM spatial resolutions, and the local coordinate origin information required to convert between image pixels and scene-local metric coordinates.

#### UAV sample NPZ files

Each `Lxx_????.npz` file stores one UAV image sample and its metadata. The dataloader reads these files through `AnyVisLocNPZDataset` in `avl_data.py`. The current dataloader returns:

| Key | Description |
| --- | --- |
| `image` | RGB UAV image tensor in `[3, H, W]`, normalized to `[0, 1]`. |
| `K` | Camera intrinsic matrix after UAV image downsampling, shape `[3, 3]`. |
| `dist` | Distortion coefficients, shape `[5]`. |
| `image_size` | Loaded/downsampled UAV image size, `[H, W]`. |
| `pose_c2w` | Camera-to-world pose matrix, shape `[4, 4]`. |
| `pose_w2c` | World-to-camera pose matrix, shape `[4, 4]`. |
| `xyz` | UAV position in the dataset-local metric coordinate system, `[x, y, z]`. |
| `euler_deg` | UAV attitude in degrees, `[roll, pitch, yaw]`. |
| `reference` | Lightweight reference pointer, including `ref_json_path`, `ref_path`, and `scene_dir`. |
| `sample_id` | Sample identifier, such as `L01_0001`. |

<!-- Run Full Pipeline -->
<a name="Run-the-Full-Pipeline"></a>

### 🏃 Run the Full Pipeline

The full pipeline runs retrieval, matching, and PnP localization in sequence.

```bash
python run_avl.py \
  --dataset_root ./Data/AnyVisLoc \
  --yaml config_selectable_matchers.yaml \
  --save_dir ./Result/AnyVisLoc \
  --reference_mode aerial \
  --pose_priori yp \
  --strategy Topn_opt \
  --PnP_method P3P \
  --match_keypoints 3000 \
  --retrieval_k 5 \
  --success_thresholds 1 3 5 10 20 \
  --device cuda
```

Run only selected scenes:

```bash
python run_avl.py \
  --dataset_root ./Data/AnyVisLoc \
  --yaml config_selectable_matchers.yaml \
  --reference_mode aerial \
  --scenes Scene_01 Scene_02 \
  --device cuda
```

Quickly test the first few samples:

```bash
python run_avl.py \
  --dataset_root ./Data/AnyVisLoc \
  --yaml config_selectable_matchers.yaml \
  --reference_mode aerial \
  --limit 20 \
  --device cuda
```

To enable visualization for qualitative inspection, add:

```bash
--visualize
```

> ⚠️ For faster evaluation, keep visualization disabled by default or explicitly use:

```bash
--no-visualize
```

The full pipeline reads `RETRIEVAL_METHODS` and `MATCHING_METHODS` from the YAML configuration file. For example, the provided `config_selectable_matchers.yaml` uses:

```yaml
RETRIEVAL_METHODS:
- CAMP

MATCHING_METHODS:
- SP_LG_GIM

PNP_METHODS:
- P3P

RETRIEVAL_COVER: 50
RETRIEVAL_TOPN: 5
RETRIEVAL_FEATURE_NORM: true
BATCH_SIZE: 128
```

To test different retrieval or matching methods in the full pipeline, edit the YAML file.

<!-- Retrieval Only -->
<a name="Run-Retrieval-Only"></a>
### 🔎 Run Retrieval Only

Use `run_avl_retrieval_only.py` to evaluate image-level retrieval without pixel matching or PnP.

```bash
python run_avl_retrieval_only.py \
  --dataset_root ./Data/AnyVisLoc \
  --yaml config_selectable_matchers.yaml \
  --save_dir ./Result/AnyVisLoc_retrieval_only \
  --reference_mode aerial \
  --retrieval_methods CAMP \
  --retrieval_ks 1 3 5 \
  --pose_priori yp \
  --device cuda
```

Evaluate satellite references instead of aerial references:

```bash
python run_avl_retrieval_only.py \
  --dataset_root ./Data/AnyVisLoc \
  --yaml config_selectable_matchers.yaml \
  --reference_mode satellite \
  --retrieval_methods CAMP \
  --retrieval_ks 1 3 5 \
  --device cuda
```

<details>
<summary><strong>Useful retrieval-only arguments</strong></summary>

| Argument | Description |
| --- | --- |
| `--retrieval_methods` | Overrides `RETRIEVAL_METHODS` in the YAML file. Example: `--retrieval_methods CAMP`. |
| `--retrieval_ks` | Sets the K values for `Recall@K` and `PDM@K`. Example: `--retrieval_ks 1 3 5`. |
| `--ref_feature_cache_dir` | Sets a custom directory for cached reference/gallery retrieval features. |
| `--disable_ref_feature_cache` | Disables the reference feature cache. |

</details>

<!-- Matching and Localization Only -->
<a name="Run-Matching-and-Localization-Only"></a>
### 🧷 Run Matching and Localization Only

Use `run_avl_match_loc.py` to evaluate pixel-level matching and PnP localization without image retrieval. In this mode, the reference patch is cropped from the ground-truth view center, making it suitable for evaluating matching and localization under oracle coarse localization.

```bash
python run_avl_match_loc.py \
  --dataset_root ./Data/AnyVisLoc \
  --yaml config_selectable_matchers.yaml \
  --save_dir ./Result/AnyVisLoc_match_loc \
  --reference_mode aerial \
  --matching_methods SP_LG_GIM \
  --pose_priori yp \
  --PnP_method P3P \
  --match_keypoints 3000 \
  --min_matches 5 \
  --device cuda
```

Test several matchers:

```bash
python run_avl_match_loc.py \
  --dataset_root ./Data/AnyVisLoc \
  --yaml config_selectable_matchers.yaml \
  --matching_methods SP_LG SP_LG_GIM ALIKED_LG DISK_LG \
  --reference_mode aerial \
  --device cuda
```

<details>
<summary><strong>Useful matching/localization-only arguments</strong></summary>

| Argument | Description |
| --- | --- |
| `--matching_methods` | Overrides `MATCHING_METHODS` in the YAML file. Supported names include `Roma`, `SP_LG`, `SP_LG_GIM`, `SP_LG_MINIMA`, `ALIKED_LG`, and `DISK_LG`. |
| `--resize_ratio` | Sets the resize ratio for UAV images before matching. |
| `--match_keypoints` | Sets the maximum number of sparse keypoints used by selectable matchers. |
| `--pnp_reproj_error` | Sets the RANSAC reprojection error threshold for PnP. |
| `--pnp_iterations` | Sets the maximum number of RANSAC iterations for PnP. |
| `--pnp_confidence` | Sets the RANSAC confidence for PnP. |

</details>

<!-- Important Arguments -->
<a name="Important-Arguments"></a>

### ⚙️ Important Arguments

Common arguments shared by the new runners:

<details>
<summary><strong>Click to expand common arguments</strong></summary>

| Argument | Options / Example | Description |
| --- | --- | --- |
| `--yaml` | `config_selectable_matchers.yaml` | Configuration file for retrieval, matching, and evaluation settings. |
| `--scenes` | `Scene_01 Scene_02` or `Scene_01,Scene_02` | Tests selected scenes only. Omit this argument to test all scenes. |
| `--strategy` | `Top1` / `Topn_opt` | Candidate-selection strategy for the full pipeline. `Top1` matches only the top retrieval candidate. `Topn_opt` is the default and evaluates the top-N retrieval candidates before selecting the best PnP result. |
| `--patch_scale` | `1.0` by default | Physical reference-patch scale. The reference patch is estimated from the UAV image footprint using altitude, camera intrinsics, and pose prior. This value multiplies that footprint before cropping the reference map. Larger values add context but increase runtime and distractors; smaller values are faster but less tolerant to pose-prior errors. |
| `--reference_mode` | `aerial` / `satellite` | Selects aerial or satellite reference maps. |
| `--pose_priori` | `yp` / `p` | `yp` uses pitch, yaw, and altitude for view-center estimation and yaw-aligns the reference map. `p` keeps the pose-aware view-center estimate but does not yaw-align the map. |
| `--limit` | `20` | Processes only the first N samples for quick debugging. `0` means all samples. |

</details>

<!-- Outputs -->
<a name="Outputs"></a>

### 📁 Outputs

The scripts save results in timestamped folders under `--save_dir`. Expand the sections below to view the typical full-pipeline layout and the meaning of each main result file.

<details>
<summary><strong>Typical full-pipeline output structure</strong></summary>

```text
Result/AnyVisLoc/
└── 20260629-1200_all/
    └── aerial/
        ├── aerial-CAMP-SP_LG_GIM-yp/
        │   ├── all_results.json
        │   ├── all_results.csv
        │   ├── all_results.jsonl
        │   ├── failed.json
        │   ├── failed.jsonl
        │   ├── summary.json
        │   ├── retrieval_metrics.json
        │   ├── retrieval_metrics_by_scene.csv
        │   ├── success_curve_overall.png
        │   ├── scene_success_stats/
        │   └── Scene_01/
        │       └── L01_0001/
        │           └── VG_data_L01_0001.pkl
        ├── all_results.json
        ├── all_results.csv
        ├── failed.json
        └── summary.json
```

</details>

<details>
<summary><strong>Main result files</strong></summary>

| File | Description |
| --- | --- |
| `summary.json` | Overall statistics, including success rate and mean/median localization error. |
| `all_results.csv` | Per-sample localization and retrieval metrics. |
| `all_results.json` / `all_results.jsonl` | Per-sample results in JSON format. |
| `failed.json` / `failed.jsonl` | Failed samples with error messages and tracebacks. |
| `retrieval_metrics.json` | Region-normalized retrieval metrics for the full pipeline. |
| `success_curve_overall.png` | Overall localization success curve under different distance thresholds. |
| `scene_success_stats/` | Per-scene success statistics and curves. |
| `VG_data_*.pkl` | Detailed per-sample full-pipeline result. |
| `result.json` | Per-sample result used by retrieval-only and matching/localization-only runners. |

</details>

<a name="test_dataset"></a>
### 🚀 Test Your Own Dataset

To test your own dataset with the AnyVisLoc workflow, convert it to the same NPZ-based structure:

```text
YourDatasetRoot/
├── Scene_01/
│   ├── L01_reference.json
│   ├── aerial_map.png
│   ├── aerial_dsm.npy
│   ├── satellite_map.png
│   ├── satellite_dsm.npy
│   ├── L01_0001.npz
│   ├── L01_0002.npz
│   └── ...
└── ...
```

Make sure that:

1. Each scene folder has a valid `Lxx_reference.json`.
2. Each UAV sample is stored as `Lxx_????.npz`.
3. UAV samples contain camera intrinsics, distortion coefficients, camera poses, local metric coordinates, and Euler angles.
4. Reference maps and DSMs are spatially aligned through the resolutions and local origins recorded in the reference JSON.
5. Scene names follow the `Scene_XX` convention if you want to use the default scene filtering.

<a name="test_approaches"></a>
### 🔆 Test Your Visual Localization Approaches

#### Test your own image retrieval model

1. Put your retrieval method under:

```text
./Retrieval_Models/your_approach/
```

2. Modify the retrieval model loader and feature extraction interface:

```text
./Retrieval_Models/multi_model_loader.py
./Retrieval_Models/feature_extract.py
```

3. Add your method name to the YAML configuration:

```yaml
RETRIEVAL_METHODS:
- CAMP
- YOUR_RETRIEVAL_METHOD
```

The retrieval method should provide a model and image transform that are compatible with the existing `retrieval_init()` and `retrieval_all_anyvisloc()` pipeline.

#### Test your own image matching model

1. Put your matcher under:

```text
./Matching_Models/your_approach/
```

2. Add an initialization function and a matching function following the existing matcher interface.

3. Register the matcher in `avl_utils.py`, especially in:

```python
matching_init()
run_pixel_match_anyvisloc()
```

4. Add your method name to the YAML configuration:

```yaml
MATCHING_METHODS:
- SP_LG_GIM
- YOUR_MATCHING_METHOD
```

The current code already supports the selectable sparse matchers listed in the **Supported Baseline Components** section above.

<!-- FAQ -->
<a name="FAQ"></a>
## ❓ FAQ

<details>
<summary><strong>Why do we need to perform image retrieval before image matching?</strong></summary>

In UAV visual localization, the reference map usually covers a much larger area than a single real-time UAV image. Running pixel-level matching directly over the entire map would create a large search space and heavy computational and storage costs. Under low-altitude oblique observation, image-level retrieval is also more robust to viewpoint changes than direct full-map pixel matching. Therefore, AnyVisLoc first uses image retrieval, also known as visual geo-localization or visual place recognition, to obtain coarse location candidates, and then applies pixel matching and PnP localization for accurate pose estimation.

</details>

<details>
<summary><strong>Why do we provide both aerial reference maps and satellite maps?</strong></summary>

The two reference modalities represent different practical settings. Aerial reference maps provide high-resolution geometry and support high-precision localization, but they require dedicated aerial data collection and reconstruction. Satellite reference maps are more broadly available and can be acquired without scene-specific aerial mapping, making them suitable for studying localization under more general reference conditions. AnyVisLoc supports systematic evaluation with both reference sources.

</details>

<details>
<summary><strong>Why is end-to-end retrieval–matching–PnP evaluation slow, and how can it be accelerated?</strong></summary>

Most of the runtime is spent on image retrieval rather than matching or PnP. The retrieval patch size and location are adapted to each UAV image according to its altitude and pose prior. Since different UAV images can have different altitude, pitch, and yaw values, the reference map may need to be cropped differently for each query. In the `yp` setting, the reference map is also yaw-rotated per query, which changes the gallery layout and usually requires reference-patch features to be extracted again while searching across the full reference map.

The `p` setting keeps the reference map unrotated and supports caching reference-gallery features when the patch geometry and tile layout remain the same; see `--ref_feature_cache_dir`. This can reduce repeated feature extraction, but full-map retrieval is still computationally demanding. For larger experiments, we recommend distributing independent scenes or queries across multiple GPUs, grouping samples with equivalent or near-equivalent crop settings, and precomputing or caching the corresponding gallery features. The released demo intentionally keeps the direct pose-conditioned retrieval formulation for reproducibility instead of introducing aggressive approximation or engineering-specific optimizations.

</details>

<details>
<summary><strong>Why are satellite-reference localization results substantially worse than aerial-reference results?</strong></summary>

This gap is expected and reflects the intrinsic difficulty of low-altitude UAV localization with satellite references. UAV and satellite images can differ greatly in viewpoint, spatial resolution, illumination, season, and acquisition time, and many objects may have changed between acquisitions. Satellite imagery also has substantially lower spatial resolution than low-altitude UAV imagery, so even visually corresponding objects may not support accurate pixel-level alignment. In addition, satellite reference images can contain local georeferencing distortions, mosaicking artifacts, blur, and non-orthorectified building lean. Finally, the public satellite DSM has a spatial resolution of approximately 30 m, which cannot support high-precision pose estimation in the same way as the higher-resolution aerial DSM. These factors are realistic challenges for low-altitude UAV visual localization with satellite references.

</details>

<a name="Citation"></a>
## Citation

Please cite the official AnyVisLoc paper in any public work that uses the dataset.

```bibtex
@InProceedings{Ye_2026_CVPR,
  author    = {Ye, Yibin and Teng, Xichao and Chen, Shuo and Liu, Leqi and
               Wang, Kun and Song, Xiaokai and Li, Zhang},
  title     = {Exploring the Best Way for UAV Visual Localization under
               Low-altitude Multi-view Observation Condition: A Benchmark},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition (CVPR) Findings},
  month     = {June},
  year      = {2026},
  pages     = {1731--1741}
}
```

<a name="License"></a>
<!-- License -->
## ⚠️ License

AnyVisLoc is released under the [**AnyVisLoc Dataset License**](docs/LICENSE_DATASET.md) for **non-commercial academic research, education, reproducibility, and internal evaluation only**.

Without prior written permission, users **may not** use the dataset commercially; redistribute, mirror, upload, or share the dataset; or publicly release any re-annotated, re-partitioned, relabeled, transformed, or substantially overlapping dataset, benchmark, subset, or derived data file based on AnyVisLoc. Users may create internal annotations, alternative splits, and derived metadata for non-commercial research, but **may not** publicly release them.

**Third-party satellite imagery, DSM/elevation products, and other third-party materials are not licensed by the AnyVisLoc authors; users must comply with the applicable third-party terms.**

## 💎 Acknowledgements

AnyVisLoc builds on and interfaces with several excellent open-source projects:

- [CAMP](https://github.com/Mabel0403/CAMP), used as the retrieval baseline.
- [RoMa](https://github.com/Parskatt/RoMa), used for robust dense feature matching.
- [LightGlue](https://github.com/cvg/LightGlue), used as the sparse local-feature matcher.
- [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork), used with LightGlue-based pipelines.
- [GIM](https://github.com/xuelunshen/gim), whose GIM-trained SuperPoint + LightGlue checkpoint is supported.
- [MINIMA](https://github.com/LSXI7/MINIMA), whose modality-invariant LightGlue checkpoint is supported.
- [ALIKED](https://github.com/Shiaoming/ALIKED), used as a lightweight keypoint and descriptor extractor.
- [DISK](https://github.com/cvlab-epfl/disk), available through the sparse-matching integration.
- [AW3D30](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm), used as the source of the public 30 m DSM product.
- Google Earth historical satellite imagery is subject to the applicable [Google Geo Guidelines](https://www.google.com/permissions/geoguidelines/) and other relevant third-party terms.

Please consult and comply with the licenses and citation requirements of all third-party projects and pretrained models.

<a name="Contact"></a>

## 📬 Contact

For questions about AnyVisLoc, dataset access, or adding your own retrieval or matching method to this benchmark, please contact [zhangli_nudt@163.com](mailto:zhangli_nudt@163.com).
