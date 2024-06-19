# Import the required libraries
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time

# Load the pre-trained YOLOv8 models
models = {
    "YOLOv8l": YOLO('yolov8l.pt'),
    "YOLOv8m": YOLO('yolov8m.pt'),
    "YOLOv8l-OIV7": YOLO('yolov8l-oiv7.pt'),
    "YOLOv8m-OIV7": YOLO('yolov8m-oiv7.pt')
}

# Streamlit app
st.title("Object Detection with Multiple YOLOv8 Models")
st.write("Upload an image and click 'Analyse Image' to detect objects with different models.")

st.sidebar.title('Object Detection Task')
st.sidebar.subheader('Detection')
st.sidebar.subheader('Test Samples')

# Function to perform object detection, return labels with confidence, and measure time
def detect_objects(model, image):
    # Start the timer
    start_time = time.time()

    # Perform object detection
    results = model(image)

    # End the timer
    end_time = time.time()

    # Extract labels and confidence scores
    names = model.names
    detections = []
    for result in results:
        for detection in result.boxes.cls:
            detections.append(names[detection.class_id])

    # Calculate the time taken
    time_taken = end_time - start_time

    return detections, time_taken


# File uploader widget
uploaded_file = st.file_uploader("Choose an image to analyze...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert the image to a format suitable for OpenCV
    image = np.array(image)

    # Button to trigger image analysis
    if st.button("Analyse Image"):
        st.write("Detecting objects...")

        # Perform object detection with each model and store the results
        all_results = {}
        for model_name, model in models.items():
            detected_labels_confidences, detection_time = detect_objects(model, image)
            all_results[model_name] = {
                "detections": detected_labels_confidences,
                "time": detection_time
            }

        # Display the results for each model
        for model_name, results in all_results.items():
            st.write(f"Results for {model_name}:")
            st.write(f"Time taken for detection: {results['time']:.2f} seconds")
            st.write("Detected Objects:")
            for label, confidence in results["detections"]:
                st.write(f'Label: {label}, Confidence: {confidence:.2f}')
            st.write("\n")
