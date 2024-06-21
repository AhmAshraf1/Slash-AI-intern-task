# Import the required libraries
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
from utils import detect_objects, load_image, draw_bounding_boxes

# Load the pre-trained YOLOv8 models
models = {
    "YOLOv8l": YOLO('yolov8l.pt'),
    "YOLOv8l-OIV7": YOLO('yolov8l-oiv7.pt'),
}

# Streamlit app
st.title("Object Detection with Multiple YOLOv8 Models")
st.write("Upload an image and click 'Analyse Image' to detect objects with different models.")

st.sidebar.title('Object Detection Task')
st.sidebar.subheader('Test Samples')


obj_detect = st.button(label="Image with Detected Objects", type="primary")
if obj_detect:
    st.switch_page("Images with Detected Objects")


# File uploader widget
uploaded_file = st.file_uploader("Choose an image to analyze...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Button to trigger image analysis
    if st.button("Analyse Image"):
        st.write("Detecting objects...")

        all_results = {}
        for model_name, model in models.items():
            detected_labels_confidences, detection_time, bounding_boxes = detect_objects(model, image)
            all_results[model_name] = {
                "detections": detected_labels_confidences,
                "time": detection_time,
                "Boxes": bounding_boxes}

        # Display the results for each model
        for model_name, results in all_results.items():
            print(f"Results for {model_name}:")
            print(f"Time taken for detection: {results['time']:.2f} seconds")
            print(f"Number of detected objects: {len(results['detections'])}\n")
            print("Detected Objects:")
            labels = []

            for label, confidence in results["detections"]:
                print(f'Label: {label}, Confidence: {confidence * 100:.2f}%')
                labels.append(label)
            print(f"Names of the components detected in the uploaded image: {labels}")
            print('\n')

            image_with_bb = image.copy()

            # Draw bounding boxes on the image for specific model
            image_with_bounding_boxes = draw_bounding_boxes(image_with_bb, all_results[model_name]["detections"],
                                                            all_results[model_name]["Boxes"])

            # Display the image with bounding boxes
            plt.title(f"Bounding Boxes for {model_name}")
            plt.imshow(image_with_bounding_boxes)
            plt.axis('off')
            plt.show()

            # clear the image after each plot
            image = image.copy()
