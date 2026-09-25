# Manuscript Specific Layout Region Detection

## 1. Project Overview

This project implements an end-to-end system for detecting and classifying layout regions in historical manuscript pages.

The system identifies five layout categories:

- header
- footer
- main_text
- side_text
- filler

The project provides a YOLO-based object detection pipeline, a command-line batch inference script, annotated output images, JSON metadata, and a Streamlit demonstration interface.

---

## 2. Objectives

- Detect layout regions in manuscript images.
- Classify detected regions into five layout classes.
- Generate bounding boxes, labels, and confidence scores.
- Support single-image and batch-folder inference.
- Generate annotated images and JSON metadata.
- Ensure bounding boxes remain within page boundaries.
- Provide a reproducible Python environment.

---

## 3. Layout Classes

| Class ID | Class Name |
|----------|------------|
| 0 | header |
| 1 | footer |
| 2 | main_text |
| 3 | side_text |
| 4 | filler |

---

## 4. Dataset

The project uses 20 manually annotated manuscript pages.

- Training images: 16
- Validation images: 4
- Annotation tool: CVAT
- Annotation format: YOLO
- Number of classes: 5

The original manuscript collection contains additional raw images. The original input images and metadata are kept unchanged.

---

## 5. Model

The project uses an Ultralytics YOLO object detection model.

Training configuration:

- Base model: YOLO11n
- Epochs: 50
- Image size: 1024
- Batch size: 4

The model predicts:

- layout class
- confidence score
- bounding box coordinates

Because the manually annotated dataset is relatively small, the current model is intended as a prototype. Additional annotated training data can improve generalization and detection accuracy.

---

## 6. Project Structure

```text
Manuscript_Reader/
│
├── app.py
├── inference.py
├── best.pt
├── classes.txt
├── requirements.txt
├── requirements-lock.txt
├── README.md
├── .gitignore
│
├── dataset/
│   ├── data.yaml
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
│
└── results/
    ├── page_31.png
    ├── page_31.json
    ├── page_33.png
    ├── page_33.json
    ├── page_37.png
    ├── page_37.json
    ├── page_40.png
    └── page_40.json