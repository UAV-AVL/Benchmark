<!--
Hey, thanks for using the awesome-readme-template template.  
If you have any enhancements, then fork this project and create a pull request 
or just open an issue with the label "enhancement".

Don't forget to give this project a star for additional support ;)
Maybe you can mention me or this repo in the acknowledgements too
-->
<div align="center">
  <video src="https://github.com/user-attachments/assets/433d9e54-46d8-42f2-9630-bf3e0c002239" controls="controls" width="500" height="300"></video>
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
   
<h4>
    <a href="https://github.com/Louis3797/awesome-readme-template/">View Paper</a>
  <span> · </span>
    <a href="https://github.com/Louis3797/awesome-readme-template">Download Dataset</a>
  <span> · </span>
    <a href="https://github.com/Louis3797/awesome-readme-template/issues/">View demo</a>

  </h4>
</div>

<br />

<!-- Table of Contents -->
# :notebook_with_decorative_cover: Table of Contents
- [Todo List](#todo)
- [The AnyVisLoc Dataset](#about-the-dataset)
  * [UAV Images Examples](#UAV-Images)
  * [Reference Map Examples](#Reference-Maps)
  * [Dataset Features](#Dataset-Features)
- [Getting Started](#toolbox-getting-started)
  * [Prerequisites](#bangbang-prerequisites)
  * [Installation](#gear-installation)
  * [Running Tests](#test_tube-running-tests)
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
  <img src="assets/reference_map1_new.png" alt="Reference Map Examples" />
</div>

<!-- Dataset Features -->
<a name="Dataset-Features"></a>
### 🌟: Dataset Features

- **Multi-altitude:** Our dataset contains low-altitude flight conditions from 30m to 300m.
- **Multi-view:**  Our dataset covers common used pitch angle of UAV imaging from 20° to 90°.
- **Multi-scene:** Our dataset includes various scenes, such as dense urban areas (e.g., cities, towns, country), typical landmark scenes (e.g., playground, museums, church), natural scenes (e.g., farmland and mountains), and mixed scenes (e.g., universities and  park).
- **Multi-reference map:** Our dataset provides two types of 2.5D reference maps for different purposes. The aerial map with high spatial resolution can be used for high-precision localization but needs pre-aerial photogrammetry. The satellite map serves as an alternative when the aerial map is unavailable.
- **Multi-drone type:** Mavic 2, Mavic 3, Mavic 3 Pro, Phantom 3, Phantom 4, Phantom 4 RTK, Mini 4 Pro
- **Others:** multiple weather(☀️⛅☁️🌫️🌧️), seasons(🌻🍀🍂⛄), illuminations(🌇🌆)


<!-- Running the baseline demo -->
## 	🚩: The baseline demo

<!-- Installation -->
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
### ⬇️: Download the dataset and the model weights

We put our dataset and the model weights on the [Baidu Netdisk](https://github.com/Mabel0403/CAMP)

<!-- Run Locally -->
### :running: Run the demo

This baseline use the [CAMP](https://github.com/Mabel0403/CAMP) model for image-level retrieval and the [Roma](https://github.com/Parskatt/RoMa) model for pixel-level matching, just run
```bash
  python baseline.py
```

if you want to test your own dataset, please put your drone images in xxx

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
