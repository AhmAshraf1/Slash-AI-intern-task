import time
import numpy as np
import cv2 as cv


def load_image(image_path):
    image = cv.imread(image_path)

    # Convert the image to a format suitable for OpenCV
    image = np.array(image)

    return image

def detect_objects(model, image):
    start_time = time.time()

    # Perform object detection
    results = model(image)

    end_time = time.time()

    # Extract labels, confidence scores and bounding boxes coordinates
    names = model.names
    detections = []
    bboxes = []
    for result in results:
        for detection in result.boxes.data:
            xmin, ymin, xmax, ymax, confidence, cls_id = detection[:6]
            cls_id = int(cls_id)
            confidence = float(confidence)
            label = names[cls_id]
            detections.append((label, confidence))
            bboxes.append((xmin, ymin, xmax, ymax))

    detection_time = end_time - start_time

    return detections, detection_time, bboxes


def draw_bounding_boxes(image, detections, boxes):
    for (label, confidence), box in zip(detections, boxes):
        # Get the bounding box coordinates
        # Convert the box tensors to a NumPy array
        box_array = np.array(box)
        x_min, y_min, x_max, y_max = box_array[0:4].astype(int)

        # Draw the bounding box
        cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Add the label and confidence score to the image
        label_text = f"{label} {confidence * 100:.2f}%"
        cv.putText(image, label_text, (x_min, y_min - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image
