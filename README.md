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
- [The Baseline Demo](#baseline)
  * [Installation](#Installation)
  * [Download Files](#Download)
  * [Run the demo](#running)
  * [Test Your Dataset](#test_dataset)
  * [Test Your Visual Localization Approaches](#test_approaches)
- [FAQ](#FAQ)
- [License](#License)
- [Acknowledgments](#Acknowledgments)
  
<!-- Roadmap -->
<a name="todo"></a>
## :compass: Todo List

* [x] Release a demo of the best combined method (Baseline) to achieve UAV visual localization.
* [x] Release 1/25 of the dataset for demo testing. (The region is an old town in Qingzhou City，China)
* [ ] Release all the UAV visual localization approaches evaluated in the benchmark.
* [ ] Release the whole dataset evaluated in the benchmark.

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
## 	🚩: The Baseline Demo

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
### ⬇️: Download Files
1. **Dataset**
  - Our dataset(1/25) is available at [Baidu Netdisk](https://pan.baidu.com/s/17U7YkFIwKcGjl-FmXmNlxg?pwd=ki5n) .
  - Please download the dataset and replace the `./Data` folder in the script.
  - Dataset are stored in the `./Data` folder like this:
```
  UAV_AVL_demo/Data/
  ├── metadata
  │   ├── test_region_name1.json
  │   ├── test_region_name2.json
  │   ├── ...
  ├── Reference_map
  │   ├── test_region_name1
  │   │   ├── aerial_2D_reference_map.tif
  │   │   ├── aerial_DSM_reference_map.tif
  │   │   ├── satellite_2D_reference_map.tif
  │   │   ├── satellite_DSM_reference_map.tif
  │   ├── test_region_name2
  │   ├──  ...
  └── UAV_image
    ├── test_region_name1
    │   ├── region1_place1
    │   │   ├── DJI_0001.JPG
    │   │   ├── DJI_0002.JPG
    │   │   ├── DJI_0003.JPG
    │   │   ├── ...
    │   ├── region1_place2
    │   │   ├── DJI_0001.JPG
    │   │   ├── DJI_0002.JPG
    │   │   ├── DJI_0003.JPG
    │   │   ├── ...
    ├── test_region_name2
    │   ├──  ...
```

2. **Model Weights**
- The model weights for image retrieval and matching are available at [CAMP](https://github.com/Mabel0403/CAMP) and [Roma](https://github.com/Parskatt/RoMa).
- We have also uploaded them on  [Baidu Netdisk](https://pan.baidu.com/s/1EqnCKiAiQfwDM7Y3LQ0QLg?pwd=q42r).
- Please download the weights and place them in the following directories:
  + For CAMP: `./Retrieval_Models/CAMP/weights/xxx.pth`
  + For RoMa: `./Matching_Models/RoMa/ckpt/xxx.pth`


<!-- Run Locally -->
<a name="running"></a>
### :running: Run the demo

This baseline use the [CAMP](https://github.com/Mabel0403/CAMP) model for image-level retrieval and the [Roma](https://github.com/Parskatt/RoMa) model for pixel-level matching, just run
```bash
  python baseline.py
```
<a name="test_dataset"></a>
### :rocket: Test your dataset
If you want to test your own dataset, please follow these steps:

1. **Prepare Drone Images**:
   - Place your drone images in the directory `.\Data\UAV_image\your_test_region`.
   - The default image format is JPG. If you use a different format (e.g., PNG), make sure to adjust the image reading function accordingly.

2. **Prepare Reference Maps**:
   - Put your reference maps in the directory `.\Data\Reference_map\your_test_region`.
   - Both the 2D reference map and the corresponding DSM (Digital Surface Model) map are required.
   - The default image format is TIF. If you use a different format, please convert it appropriately.

3. **Configure Metadata**:
   - Put your drone metadata in `.\Data\metadata\your_test_region.json`.
   - Ensure that this JSON file includes all necessary information (e.g., image path, drone 6 DoF pose and camera intrinsics).
   - Put the reference map information of your test region in `.\Regions_params\your_test_region.yaml`.
   - Ensure that this YAML file includes all necessary information (e.g., image path, spatial resolution, the WGS84 UTM system of test region and the initial coordinates of reference maps).
   - Please note that the parameters named `xxx_REF_COORDINATE` and `xxx_DSM_COORDINATE`  are used to align the 2D reference map with the DSM map. You can use geographic information software such as [ENVI](https://www.nv5geospatialsoftware.com/Products/ENVI) to open both TIF images simultaneously and select the pixel coordinates of any corresponding points as inputs for these parameters.

<a name="test_approaches"></a>
### 🔆: Test Your Visual Localization Approaches

1. **Test Your Own Image Retrieval Model**:
  - **Place Your Folder:** Put your main folder at the  `./Retrieval_Models/your_approach`.
  - **Modify Files:** Update the following files:
    + `multi_model_loader.py`: Contains the function calls for image retrieval methods.
    + `feature_extract.py`: Contains the functions for network feature processing.
    + `config.yaml`: Add the name of your image retrieval method.
  - **Suggestion:** Refer to the functions we have provided for corresponding modifications.
2. **Test Your Own Image Matching Model**:
  - **Place Your Folder:** Put your main folder at the `./Matching_Models/your_approach`.
  - **Modify Files:** Update the following files:
    + Add a `xxx_match.py` File: Include the model initialization function `xxx_Init()` and the image matching function `xxx_match()`.
Refer to our provided Roma_match.py for modifications.
    + Modify `utils.py`: Update the `matching_init()` function and the `Match2Pos_all()` function.
Add the invocation module for your method within these functions.
    + `config.yaml`: Add the name of your image matching method.

<!-- FAQ -->
<a name="FAQ"></a>
## :grey_question: FAQ

- **Why do we need to perform image retrieval before image matching?**

  + In UAV visual localization tasks, the reference map's coverage area is often much larger than the real-time images captured by the UAV. Directly applying pixel-level matching algorithms in such scenarios would lead to a massive search space and significant computational and storage pressures. Additionally, under low-altitude oblique observation conditions, image-level retrieval exhibits better robustness to viewpoint differences compared to pixel-level matching. Therefore, we recommend first using image-level retrieval (also known as visual geo-localization or visual place recognition) to find the approximate location of the UAV image, and then performing pixel-level matching.
    
- **Why do we provide both aerial photogrammetry reference maps and satellite maps?**

  + These two types of reference maps have different advantages and disadvantages. Aerial maps offer superior localization accuracy but are more cumbersome to produce. They require pre-aerial photography and precise 3D modeling of the flight area, making them less suitable for time-sensitive missions (e.g., emergency rescue) or long-distance flight tasks. Therefore, the type of reference map should be chosen based on the actual mission requirements. Our dataset supports researchers in comprehensively evaluating their localization approaches with different reference maps.

<a name="License"></a>
<!-- License -->
## :warning: License

See LICENSE.txt for more information.

<a name="Acknowledgments"></a>
<!-- Acknowledgments -->
## :gem: Acknowledgements

Use this section to mention useful resources and libraries that you have used in your projects.

 - [CAMP: A Cross-View Geo-Localization Method using Contrastive Attributes Mining and Position-aware Partitioning](https://github.com/Mabel0403/CAMP)
 - [Roma: Robust Dense Feature Matching](https://github.com/Parskatt/RoMa)
 - [ALOS 30m DSM](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30\_e.htm)
