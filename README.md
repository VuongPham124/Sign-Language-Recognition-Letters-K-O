# Sign Language Recognition — Letters K to O

Real-time ASL (American Sign Language) hand gesture recognition system using deep learning and MediaPipe, deployed via a Streamlit web app.

---

## Overview

This project trains and compares multiple deep learning models to classify ASL hand gestures for letters **K, L, M, N, O**. MediaPipe is used for hand detection and cropping, and the final system runs in real-time through a webcam-based Streamlit interface.

---

## Architecture

```
Webcam / Image Input
        ↓
MediaPipe Hand Detection (landmark + bounding box crop)
        ↓
Preprocessing (resize, normalize)
        ↓
Deep Learning Model (CustomCNN / MobileNetV2 / ResNet50)
        ↓
Prediction → Streamlit Dashboard (real-time display + log)
```

---

## Models

| Model | Input Size | Notes |
|---|---|---|
| Custom CNN | 192 × 192 | Lightweight, trained from scratch |
| MobileNetV2 | 224 × 224 | Transfer learning |
| ResNet50 | 224 × 224 | Transfer learning |

All models are available for download (see below).

---

## Features

- **Hand Detection:** MediaPipe extracts hand landmarks and crops the hand region with padding for clean input
- **Multi-model Comparison:** Three architectures trained and evaluated on the same dataset
- **Real-time Inference:** Live webcam feed with bounding box visualization and prediction overlay
- **Prediction Log:** Timestamped prediction history displayed in the app

---

## Project Structure

```
├── CustomCNN.ipynb       # Custom CNN training & evaluation
├── MobileNetV2.ipynb     # MobileNetV2 training & evaluation
├── Resnet50.ipynb        # ResNet50 training & evaluation
├── app.py                # Streamlit real-time inference app
```

---

## Setup

### 1. Install dependencies

```bash
pip install streamlit opencv-python mediapipe tensorflow numpy
```

### 2. Download trained models

- [CustomCNN](https://drive.google.com/file/d/1OOKlEqVtsgLr-rFGPMIrH3qFH-XDbNdp/view?usp=sharing)
- [MobileNetV2](https://drive.google.com/file/d/1EKae5Q8vxI8bD1QEZncBHbpy4S_wqyKP/view?usp=sharing)
- [ResNet50](https://drive.google.com/file/d/1KRijLUVUZHmHB18-qZKUXL3MDsk2cl1x/view?usp=sharing)

Place the downloaded `.h5` file in the project root directory.

### 3. Run the app

```bash
streamlit run app.py
```

To switch models, uncomment the corresponding line in `app.py`:

```python
model = load_model("customcnn_model.h5");     input_size = (192, 192)
# model = load_model("mobilenetv2_model.h5"); input_size = (224, 224)
# model = load_model("resnet50_model.h5");    input_size = (224, 224)
```

---

## Tech Stack

- **Language:** Python
- **Hand Detection:** MediaPipe
- **Modeling:** TensorFlow / Keras (Custom CNN, MobileNetV2, ResNet50)
- **App:** Streamlit, OpenCV
