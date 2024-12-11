<!--
Hey, thanks for using the awesome-readme-template template.  
If you have any enhancements, then fork this project and create a pull request 
or just open an issue with the label "enhancement".

Don't forget to give this project a star for additional support ;)
Maybe you can mention me or this repo in the acknowledgements too
-->
<div align="center">
  <video src="https://github.com/user-attachments/assets/396b94f9-3f8c-43fe-a358-17a73b812e14" controls="controls" width="500" height="300"></video>
  <!--<img src="overview.png" alt="logo" width="400" height="auto" />-->
  <h1>The First Large-scale Benchmark for UAV Visual Localization under Low-altitude Multi-view Observation Condition</h1>
  
  <p>
    This benchmark focuses on UAV visual localization under Low-altitude Multi-view observation condition using the 2.5D aerial or satellite reference maps. The visual localization is mainly achieved via a unified framework combining image retrieval, image matching, and PnP problem solving. 
  </p>
  
  
<!-- Badges
<p>
  <a href="https://github.com/Louis3797/awesome-readme-template/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/Louis3797/awesome-readme-template" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/Louis3797/awesome-readme-template" alt="last update" />
  </a>
  <a href="https://github.com/Louis3797/awesome-readme-template/network/members">
    <img src="https://img.shields.io/github/forks/Louis3797/awesome-readme-template" alt="forks" />
  </a>
  <a href="https://github.com/Louis3797/awesome-readme-template/stargazers">
    <img src="https://img.shields.io/github/stars/Louis3797/awesome-readme-template" alt="stars" />
  </a>
  <a href="https://github.com/Louis3797/awesome-readme-template/issues/">
    <img src="https://img.shields.io/github/issues/Louis3797/awesome-readme-template" alt="open issues" />
  </a>
  <a href="https://github.com/Louis3797/awesome-readme-template/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/Louis3797/awesome-readme-template.svg" alt="license" />
  </a>-->
</p> 
   
<!--<h4>
    <a href="https://github.com/Louis3797/awesome-readme-template/">View Paper</a>
  <span> · </span>
    <a href="https://github.com/Louis3797/awesome-readme-template">Download Dataset</a>
  <span> · </span>
    <a href="https://github.com/Louis3797/awesome-readme-template/issues/">View demo</a>

  </h4>-->
</div>

<br />

<!-- Table of Contents -->
# :notebook_with_decorative_cover: Table of Contents
- [Todo List](#todo)
- [The AnyVisLoc Dataset](#about-the-dataset)
  * [UAV Images Examples](#UAV-Images)
  * [Reference Map Examples](#Reference-Maps)
  * [Dataset Features](#Dataset-Features)
- [The baseline demo](#baseline)
  * [Prerequisites](#bangbang-prerequisites)
  * [Installation](#Installation)
  * [Download the dataset and the model weights](#Download)
  * [Run Locally](#running-run-locally)
  * [Deployment](#triangular_flag_on_post-deployment)

<!-- Roadmap -->
<a name="todo"></a>
## :compass: Todo List

* [x] Release a demo of the best combined method (Baseline) to achieve UAV visual localization.
* [ ] Release all the UAV visual localization approaches evaluated in the benchmark.  

<!-- About the AnyVisLoc Dataset -->
<a name="about-the-dataset"></a>
## 📸: The AnyVisLoc Dataset: First Large-scale Low-altitude Multi-view UAV AVL dataset


<!-- UAV Images Examples -->
<a name="UAV-Images"></a>
### ✈️: UAV Images

<div align="center"> 
  <img src="assets/overview_supp.png" alt="UAV Image Examples" />
</div>

<!-- Reference Map -->
<a name="Reference-Maps"></a>
### 🗺️: Reference Maps

<div align="center"> 
  <img src="assets/reference_map_new1.png" alt="Reference Map Examples" />
</div>

<!-- Dataset Features -->
<a name="Dataset-Features"></a>
### 🌟: Dataset Features
- **Large scale:** **18,000** full-resolution DJI images taken from **15** different cities across China. The reference maps cover **25** distinct regions ranging in coverage area from **10,000 $m^2$ to 9,000,000 $m^2$**.
- **Multi-altitude:** The dataset contains low-altitude flight conditions from **30m to 300m**.
- **Multi-view:**  The dataset covers common used pitch angle of UAV imaging from **20° to 90°**.
- **Multi-scene:** The dataset includes various scenes, such as dense **urban** areas (e.g., cities, towns, country), typical **landmark** scenes (e.g., playground, museums, church), **natural** scenes (e.g., farmland and mountains), and **mixed** scenes (e.g., universities and  park).
- **Multi-reference map:** The dataset provides two types of 2.5D reference maps for different purposes. The **aerial map** with high spatial resolution can be used for high-precision localization but needs pre-aerial photogrammetry. The **satellite map** serves as an alternative when the aerial map is unavailable.
- **Multi-drone type:** Mavic 2, Mavic 3, Mavic 3 Pro, Phantom 3, Phantom 4, Phantom 4 RTK, Mini 4 Pro
- **Others:** multiple weather(☀️⛅☁️🌫️🌧️), seasons(🌻🍀🍂⛄), illuminations(🌇🌆)


<!-- Running the baseline demo -->
<a name="baseline"></a>
## 	🚩: The baseline demo

<!-- Installation -->
<a name="Installation"></a>
### :gear: Installation
Clone the project

```bash
  git clone https://github.com/UAV-AVL/Benchmark.git
```

Install dependencies(tested on windows python 3.9)

```bash
  pip install -r requriements.txt
```
   
<!-- Download-->
<a name="Download"></a>
### ⬇️: Download the dataset and the model weights
1. **Dataset**
  - Our dataset(1/25) is available at [Baidu Netdisk](https://pan.baidu.com/s/17U7YkFIwKcGjl-FmXmNlxg?pwd=ki5n) .
  - Please download the dataset and replace the `./Data` folder in the script.
1. **Model Weights**
- The model weights for image retrieval and matching are available at [CAMP](https://github.com/Mabel0403/CAMP) and [Roma](https://github.com/Parskatt/RoMa).
- We have also uploaded them on  [Baidu Netdisk](https://pan.baidu.com/s/1EqnCKiAiQfwDM7Y3LQ0QLg?pwd=q42r).
- Please download the weights and place them in the following directories:
  + For CAMP: `./Retrieval_Models/CAMP/weights/xxx.pth`
  + For RoMa: `./Matching_Models/RoMa/ckpt/xxx.pth`
<!-- Run Locally -->
### :running: Run the demo

This baseline use the [CAMP](https://github.com/Mabel0403/CAMP) model for image-level retrieval and the [Roma](https://github.com/Parskatt/RoMa) model for pixel-level matching, just run
```bash
  python baseline.py
```
### :rocket: Test your own dataset
If you want to test your own dataset, please follow these steps:

1. **Prepare Drone Images**:
   - Place your drone images in the directory `.\Data\UAV_image\your_test_region`.
   - The default image format is JPG. If you are using other format (e.g., PNG), make sure to adjust the image reading function accordingly.

2. **Prepare Reference Maps**:
   - Put your reference maps in the directory `.\Data\Reference_map\your_test_region`.
   - Both the 2D reference map and the corresponding DSM (Digital Surface Model) map are required.
   - The default image format is TIF. If you use a different format, please convert it appropriately.

3. **Configure Metadata**:
   - Put your drone metadata in `.\Data\metadata\your_test_region.json`.
   - Ensure that this JSON file includes all necessary metadata information, including the image path, drone 6 DoF pose(ground truth) and camera intrinsics.

```bash
  cd my-project
```
and put your reference maps in xxx, the dataset configuration files have to be modified too.

```bash
  yarn install
```

Start the server

```bash
  yarn start
```

Please read the [Code of Conduct](https://github.com/Louis3797/awesome-readme-template/blob/master/CODE_OF_CONDUCT.md)

<!-- FAQ -->
## :grey_question: FAQ

- Question 1

  + Answer 1

- Question 2

  + Answer 2


<!-- License -->
## :warning: License

Distributed under the no License. See LICENSE.txt for more information.


<!-- Acknowledgments -->
## :gem: Acknowledgements

Use this section to mention useful resources and libraries that you have used in your projects.

 - [CAMP: A Cross-View Geo-Localization Method using Contrastive Attributes Mining and Position-aware Partitioning](https://github.com/Mabel0403/CAMP)
 - [Roma: Robust Dense Feature Matching](https://github.com/Parskatt/RoMa)
 - [ALOS 30m DSM](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30\_e.htm)
