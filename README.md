# Retinal Vessel Segmentation using U-Net Architectures

This repository contains a deep learning project for segmenting blood vessels in retinal fundus images using U-Net and Res-U-Net. This task is crucial for the early diagnosis of various ophthalmological diseases, such as diabetic retinopathy.

---

## Project Overview
The goal of this project is to accurately identify and segment the intricate network of blood vessels from a retinal image. The methodology involves:
1.  **Preprocessing:** Enhancing image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization) on the green channel of the fundus images, which typically offers the best vessel visibility.
2.  **Patching:** Dividing large retinal images into smaller patches to make them suitable for training a deep neural network with limited memory.
3.  **Modeling:** Implementing and training U-Net and Residual U-Net models for the segmentation task.
4.  **Evaluation:** Assessing model performance using key metrics like Accuracy and Mean Intersection over Union (IoU).

---

## Features
- **Multiple Model Architectures:** Implements a standard **U-Net** and a **Residual U-Net** (`Res-Unet`) for improved feature learning.
- **Image Preprocessing:** Utilizes the green channel of retinal images and applies **CLAHE** for robust contrast enhancement.
- **Patch-Based Training & Inference:** Employs the `patchify` library to handle large medical images efficiently.

---

## Model Architectures
The `model.py` file defines the neural network architectures used in this project.

1.  **U-Net (`unetmodel`)**: A classic convolutional neural network architecture designed for biomedical image segmentation. Its U-shaped structure consists of an encoder (downsampling path) to capture context and a decoder (upsampling path) with skip connections to enable precise localization.

2.  **Residual U-Net (`residualunet`)**: An enhancement of the standard U-Net that replaces the standard convolutional blocks with residual blocks. These blocks use skip connections within themselves to help mitigate the vanishing gradient problem, allowing for deeper and more effective networks.

---

## Dataset
The datasets of the fundus images can be acquired from:

HRF
DRIVE
STARE
---


## Results
This section showcases the performance of the trained models on the test dataset.

### Quantitative Results

| Model          | Average Accuracy | Mean IoU  |
| -------------- | ---------------- | --------- |
| U-Net          | **95.49%**       | **0.751** |
| Residual U-Net | **95.70%**       | **0.763** |


#### Segmentation Examples

Below is a sample prediction from the test set, showing the original image, the preprocessed input, the ground truth mask, and the mask predicted by the model.

<img width="6000" height="2000" alt="test_image_4views" src="https://github.com/user-attachments/assets/fd2605d5-a52e-4d01-927f-3e3ec7b5a82d" />

---

## Installation
To set up the environment and run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Rohit-Gupta31/retinal-vessel-segmentation.git](https://github.com/Rohit-Gupta31/retinal-vessel-segmentation.git)
    cd retinal-vessel-segmentation
    ```

2.  **Create a `requirements.txt` file** with the following content:
    ```
    tensorflow
    scikit-learn
    opencv-python
    numpy
    matplotlib
    patchify
    Pillow
    scikit-image
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage
The project is divided into two main scripts: `train.py` for training the model and `test.py` for evaluating it.

### 1. Training the Model

The `train.py` script handles data loading, preprocessing, and model training.

-   **To select a model**, open `train.py` and uncomment the desired model definition and compilation block.
-   **To start training**, run the script from the terminal:
    ```bash
    python train.py
    ```
### 2. Testing the Model

The `test.py` script is used to perform inference on the test set and calculate final performance metrics.

-   **Before running**, open `test.py` and ensure the correct model architecture is uncommented and the `model.load_weights()` path points to your saved `.keras` file.
-   **To start testing**, run the script:
    ```bash
    python test.py
    ```
---
