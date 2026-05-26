import cv2
import sys
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import json

from config import *

# Pre-defined bird class ID in the COCO dataset is 14
BIRD_CLASS_ID = 14


class BirdDetectionApp:
    def __init__(self, clf_model_path, yolo_model_path, labels_path):
        # Load YOLO detector
        self.detector = YOLO(yolo_model_path)

        # Load custom TensorFlow classifier
        self.classifier = tf.keras.models.load_model(clf_model_path)

        # --- Load the labels JSON ---
        with open(labels_path, 'r') as f:
            self.class_labels = json.load(f)

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. Run object detection (filter exclusively for birds)
        results = self.detector(img, classes=[BIRD_CLASS_ID])


        result = results[0]
        box = result.boxes[0]

        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop the bird from the image
        bird_crop = img_rgb[y1:y2, x1:x2]
        if bird_crop.size == 0:
            print("No bird detected")
            return -1

        # 2. Preprocess crop for classification
        bird_resized = cv2.resize(bird_crop, IMG_SIZE)
        bird_array = tf.keras.preprocessing.image.img_to_array(bird_resized)
        bird_array = tf.expand_dims(bird_array, 0)  # Create batch axis
        bird_preprocessed = tf.keras.applications.mobilenet_v3.preprocess_input(bird_array)

        # 3. Classify bird species
        predictions = self.classifier.predict(bird_preprocessed)
        top_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        # 4. Draw bounding box and label
        # --- Map the index to the JSON label ---
        species_name = self.class_labels[top_class_idx]
        label = f"{species_name} ({confidence:.2f})"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Define a maximum width or height (e.g., 800 pixels)
        max_dim = 1600
        h, w = img.shape[:2]
        if w > max_dim or h > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        # Show image
        cv2.imshow("Bird Detection & Classification", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = BirdDetectionApp(MODEL_SAVE_PATH, YOLO_MODEL_PATH, LABELS_PATH)

    while True:
        print("\n" + "*" * 45)
        print("* Swedish Bird Classifier & Detector App  *")
        print("*" * 45)

        choice = input("\nEnter filename (example: fiskmas.jpg) or type exit to quit: ").strip()

        if choice == 'exit':
            print("Exiting application. Goodbye!")
            sys.exit(0)
        else:
            app.process_image(TEST_IMAGES_PATH + choice)
