# ThermSplit-GS
Code available for "ThermSplit-GS: Distortion-Aware Thermal Gaussian Splatting for Robust Infrared-Only Scene Reconstruction"

## Method Overview
![ThermSplit-GS](images/method.png)

**Figure 1:** Overview of ThermSplit-GS. The framework contains two complementary components: CDDM, which approximates capture-side disturbance during representation learning, and GC-TCE, which refines rendered thermal maps by alleviating conduction-related degradations. The pipeline starts with camera pose estimation and sparse point cloud initialization from input infrared images, followed by disturbance-aware optimization in CDDM and geometry-aware refinement for the final output

## ✅ Completed
1.Integrated camera pose forward/backward propagation into the Gaussian rasterization pipeline.
2.Released scene and 3D Gaussian model initialization code.


## 🔄 To Do
Upload and organize training/testing scripts.
