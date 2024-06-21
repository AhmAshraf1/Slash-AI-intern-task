# Object Detection with YOLOv8 Model
A Streamlit web app for object detection using YOLOv8 model. The app allows users to upload image and shows detection 
results and time taken for detection with drawn bounding boxes on the uploaded image.

## Table of contents:
- [Overview & Features](#overview)
- [Requirements](#requirements)
- [Structure](#structure)
- [Application](#application)
- [Demo](#demo)
- [Models](#models)

## Overview
Using two versions of YOLOv8l model trained on two different datasets, [COCO 2017 Dataset](#https://cocodataset.org/#home)
and [Open Images V7 Dataset](#https://docs.ultralytics.com/datasets/detect/open-images-v7/), to apply object detection.

### **Features:**
- Time Detection: which is the time each model takes for detecting different components in the image measured in seconds.
- Number of Detected objects.
- Labels and confidence : shows each detected component label with model's confidence.
- List that contains names of detected componentes in the image.
- Uploaded Image with Bounding boxes drawn around each detected component.

## Requirements
- **Install dependencies & requirements**:
    ```bash
    pip install -r requirements.txt
    ```

- **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```
  
## Structure
- Object-Detection-Task.ipynb: Notebook where I tested different models before deployment
- yolov8l.pt: file of YOLOv8 model with size large trained on COCO dataset.
- yolob8l-oiv7.pt: file of YOLOv8 model with size large trained on Open Images V7 dataset.
- utils.py: functions that I used for image processing and detection
- app.py: main application file for streamlit web app.


## Application
A streamlit web application to make a friendly user interface to be more easily for user.

you can access the web app from below.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_red.svg)](https://object-detection-task.streamlit.app/)

- The Web app asks you to upload an image for object detection, then it shows the uploaded image.
- you need to click Analyse image button to start object detection, It shows the results with different [features](#features)

## Demo


## Models 
- Here is the link I used to download YOLOv8 model, YOLOv8l & YOLOv8m,trained on COCO 2017 Dataset [Models Link](https://github.com/ultralytics/ultralytics/blob/main/docs/en/tasks/detect.md)
- The other two models, YOLOv8l-oiv7 & YOLOv8m-oiv7, I downloaded was trained on Open Images V7 Dataset [Models link](https://docs.ultralytics.com/datasets/detect/open-images-v7/)

