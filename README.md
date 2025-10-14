# Towards Robust Brain Midline Shift Detection: A YOLO-Based 3D Slicer Extension with a Novel Dataset

This repository contains the **"brain-midline-detection-dataset"** and source code accompanying the paper:

If you use this dataset or code, please cite:

Kurt Pehlivanoğlu, M., Albayrak, N.B., Karhan, D. et al. Towards Robust Brain Midline Shift Detection: A YOLO-Based 3D Slicer Extension with a Novel Dataset. Neuroinform 23, 50 (2025). https://doi.org/10.1007/s12021-025-09748-z

📄 [Read the paper](https://doi.org/10.1007/s12021-025-09748-z)

This repository provides:
- A **novel MRI-based dataset** for detecting three anatomical brain landmarks:
  - **AF (Anterior Falx)**
  - **PF (Posterior Falx)**
  - **SP (Septum Pellucidum)**
- A **YOLOv5m-based deep learning model** optimized for brain midline shift detection.
- An **integrated 3D Slicer extension** for automatic, real-time detection and visualization of midline shift in brain MRIs.




## Table of Contents

* [Overview](#overview)
* [Dataset Details](#dataset-details)
* [Data Preprocessing Stages](#data-preprocessing-stages)
    - [Separating Tissues Outside the Brain](#separating-tissues-outside-the-brain)
    - [Normalization and Transformation](#normalization-and-transformation)
* [Additional Steps](#additional-steps)
    - [Preparing the Dataset](#preparing-the-dataset)
    - [Data Augmentation](#data-augmentation)
* [3D Slicer Integration](#3d-slicer-integration)
    - [Model Selection and Preparation](#model-selection-and-preparation)
    - [Integration Steps](#integration-steps)
    - [Application Overview](#application-overview)
* [Clinical Validation](#clinical-validation)
    - [Comparative outputs on the ReMIND dataset](#comparative-outputs-on-the-remind-dataset)
    - [Comparative outputs on the BraTS dataset](#comparative-outputs-on-the-brats-dataset)
* [Additional Models and Files](#additional-models-and-files)


## **Overview**

This study aims to calculate brain midline shift. Brain midline shift can occur due to traumatic or natural causes such as hematomas, tumors, and intracranial cysts. Shifts greater than 5 mm often require urgent surgical intervention. This shift is usually measured as the displacement of the septum pellucidum (SP) region between the anterior falx (AF) and posterior falx (PF). Existing models in the literature typically calculate the shift using pre-acquired MRI and CT images. However, in this study, a new module has been developed using brain MRI images and a patient-specific brain model. Deep learning methods have been used to detect the AF, PF, and SP regions.

<p align="center"> <img width="600" src="imgs/image.png"> </p>

The method focuses on the axial plane of the 3D MRI images, analyzing sections in this plane. The amount of shift is calculated and visualized using mathematical operations. As a result, an MRI file showing the amount of brain midline shift is created by combining the shifts in the 2D plane. This developed module provides rapid and accurate shift detection in clinical decision support systems, with the potential to improve patient outcomes through early diagnosis.

``` 
📌 The module integrates with the 3D Slicer platform to detect and visualize brain midline shifts. This integration enhances the precision and effectiveness of brain surgery planning and evaluation processes.
```

<p align="center"> <img width="1000" src="imgs/execute_map.png"> </p>

---

### **Dataset Details**

🗄️ The Brain Resection Multimodal Imaging Database (ReMIND) provided by The Cancer Imaging Archive has been chosen. This dataset includes preoperative MRI images, intraoperative ultrasound images, intraoperative MRI images, and segmentation series. Here are the dataset details:

```
🔸 369 preoperative MRI images
🔸 320 3D intraoperative ultrasound images
🔸 301 intraoperative MRI images
🔸 356 segmentation series
```

❗️ These data were collected from 123 different patients. However, due to image quality issues, data from 9 patients were excluded, resulting in 114 patient data being used.

For more information, detailed data about the dataset can be found on [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/).

### **Data Preprocessing Stages**

#### **Separating Tissues Outside the Brain**

⚙️ This is performed using a plugin within the 3D Slicer software. Using an atlas mask, non-brain tissues are separated with a level set algorithm, obtaining only the brain image.
MRI images are processed with `Swiss Skull Stripper`, and then non-brain tissues are separated using an atlas mask.

<p align="center"> <img width="600" src="imgs/Skull_stripper.png"> </p>

#### Normalization and Transformation

Images are read, normalized, and converted to grayscale.

- **Parameters:**
``` 
🔹 Alpha value: 0
🔹 Beta value: 256
🔹 Normalization type: norm-minmax
🔹 Data type: unsigned 8-bit
``` 
Then, they are rotated 90 degrees clockwise and flipped along the vertical axis to align correctly for the model to produce accurate outputs.

---

### Additional Steps

### Preparing the Dataset

The labeled dataset and usable models have been made publicly available on the Roboflow platform under the name [brain-midline-detection-dataset](https://universe.roboflow.com/brain-point-detection/brain-midline-detection-dataset/model/5). This dataset includes 717 axial MRI images labeled with the coordinates of AF, PF, and SP points.

<p align="center">
    <img width="400" src="imgs/af_pf_sp.png">    
</p>

<p align="center">
    <img src="imgs/roboflow.png">
</p>

### Data Augmentation

To increase the robustness of the model, data augmentation techniques have been applied to the labeled dataset. These techniques include:

```
♦️ Rotation: Images were rotated between -5° and +5°.
♦️ Brightness Adjustment: Brightness levels were changed by ±15%.
♦️ Blurring: Images were blurred up to 1 pixel.
```

<p align="center"> <img width="700" src="imgs/augmentation.png"> </p>

```
↪️ As a result, the dataset has been expanded to a total of 1,454 images. The augmented dataset is divided into training, validation, and test sets with the following distribution:

➟ Training Set: 1,230 images (85%)
➟ Validation Set: 158 images (10%)
➟ Test Set: 66 images (5%)
```


### **3D Slicer Integration**

<p align="center"> <img src="imgs/3DSlicer.png"> </p>

#### Model Selection and Preparation
- **YOLOv5m**: Selected for integration due to its compatibility and strong performance. This model has been specially trained on datasets to effectively detect AF, PF, and SP points.
- **YOLOv8 and YOLOv9**: These models have also been trained for comparative analysis. Although improvements in results and performance were achieved, YOLOv5m model was preferred due to integration incompatibility.

#### Integration Steps
- **Preprocessing**: MRI images were processed using OpenCV and Nibabel libraries. These images were read, converted to grayscale, and normalized.
- **Transformation**: To ensure correct alignment for detection, images were rotated 90 degrees counterclockwise.
- **Detection and Storage**: Detected points were stored and saved as JSON files compatible with 3D Slicer.
- **Visualization**: Outputs were visualized in 3D Slicer, providing a comprehensive view of the detected points and their spatial relationships.

<img width="1920" height="1080" alt="Anterior" src="https://github.com/user-attachments/assets/5edc646c-bbfc-4fea-9ea7-5654b487f956" />

#### Application Overview
- **Nibabel Library**: Used to read MRI images.
- **OpenCV Library**: Used for image processing, including reading axial section images, converting them to grayscale, and normalizing them.
- **Model Rotation**: Applied to correctly align images for better detection accuracy.
- **Creating JSON Files**: Used to store detected points and save them as JSON files compatible with 3D Slicer for visualization.

<p align="center"> <img width="480" height="480" alt="Screenshot_5" src="https://github.com/user-attachments/assets/f1856863-d8fb-4df5-a9ab-836af0777a6c" /> </p>



---

## **Clinical Validation**

To evaluate the model’s performance in a clinical context, predicted and ground-truth (GT) midline shift values are compared for selected patients from the ReMIND and BraTS datasets.
The predicted shift is calculated as the perpendicular Euclidean distance between the SP point identified by YOLOv5m and the predicted ideal midline, which is defined using the AF and PF points also predicted by YOLOv5m.
Likewise, the GT shift is calculated as the perpendicular Euclidean distance between the SP point annotated by an expert surgeon and the GT ideal midline, determined according to the expert-annotated AF and PF points.
This validation procedure provides a reliable assessment of the model’s accuracy and its potential clinical applicability.
The following images provide an overview of the landmark detection performance and centerline shift calculation for 5 patients in the BraTS dataset and 7 patients in the ReMIND dataset.


### **Comparative outputs on the ReMIND dataset**
Experiments on the ReMIND dataset are conducted using data from seven randomly selected patients (IDs: 20, 24, 25, 58, 72, 103, and 113). As noted earlier, a confidence
threshold of 0.7 is applied to the AF, SP, and PF detections in each slice. Consequently, nine slices were excluded from the experiments as their confidence scores fell below the threshold set by the YOLOv5m model.

<p align="center"> <img width="800" alt="brain_shift_images_REMIND_new" src="https://github.com/user-attachments/assets/a9497cdd-c819-4a78-993e-937e4f5a0421" /> </p>

### **Comparative outputs on the BraTS dataset**
To assess the generalizability of the system, experiments are conducted on the BraTS 2024 dataset using data from five randomly selected patients (IDs: 106, 107, 242, 937, and 1155).
As previously mentioned, a confidence threshold of 0.7 was applied to the AF, SP, and PF detections on each slice.
Consequently, 17 slices were excluded from the experiments because their confidence scores fell below the threshold set by the YOLOv5m model.

<p align="center"> <img width="800" alt="brain_shift_images-BrATS" src="https://github.com/user-attachments/assets/bbc40d4b-ab1a-4c4c-afab-05c98cf66ab3" /> </p>

---

## Additional Models and Files
🛠️ Additional models can be included to enhance the functionality and versatility of the 3D Slicer extension. These models should be extensively tested and documented to ensure seamless integration and provide comprehensive usage guides. Detailed descriptions, selection criteria, and integration steps for each model should be provided.

By following these guidelines, the 3D Slicer extension has been developed to provide users with practical tools for detecting brain midline shift. Developing comprehensive models and adding MRI image filters specific to the application will ensure the application is user-friendly and widely adopted.
