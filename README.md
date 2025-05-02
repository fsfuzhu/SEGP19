# Cell Recognition in Pap Smear Whole Slide Images: A Deep Learning Approach

## Project Overview

This project aims to improve the efficiency of pap smear screenings by automating the detection and classification of cells in Whole Slide Images (WSIs). By leveraging deep learning techniques, we've developed a two-stage system that:

1. Detects individual cells within WSIs using YOLOv8
2. Classifies each detected cell as normal or abnormal using ResNet18

This approach helps pathologists by reducing the time-consuming manual examination of samples while maintaining diagnostic accuracy.

## Architecture

The system employs a two-stage pipeline:

- **Cell Detection**: YOLOv8-based object detection model identifies and localizes individual cells within whole slide images
- **Cell Classification**: ResNet18-based classification model categorizes detected cells
- **User Interface**: Python-based application with CustomTkinter for easy use by pathologists

## Main Directory Structure

- `model_src/`: Source code for the models
  - `model_src/RESNET18/`: Classification model source code and configurations
  - `model_src/yolo/`: Object detection model source code
- `tools/`: Auxiliary tools for data processing
  - `tools/pathologist_verification/`: Tools for model validation by pathologists
  - `tools/split_svs_to_jpg/`: Utilities for converting SVS files to JPG format

## Setup and Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Required libraries: torch, torchvision, ultralytics, opencv-python, numpy, customtkinter

### Installation
1. Clone this repository
2. Install required dependencies:
   ```
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124
   pip install opencv-python numpy ultralytics customtkinter
   ```

### Model Performance
- Cell Detection (YOLOv8): 92.5% mAP@0.5, 91.2% precision, 94.3% recall
- Cell Classification (ResNet18): 96.36% precision, 98.63% recall, 97.48% F1 score

## Project Team

This project was developed by Group 19 at the University of Nottingham Malaysia:
- Chea Yong Zen
- Seow Weng Yann
- Wang Jun Ru
- Qiu Yi Xuan
- Jolene Swee Jung Yen

Supervisor: Mr Chew Sze Ker

## Acknowledgements

Special thanks to Cytovision Sdn Bhd for providing data and domain expertise for this project.
