import os

# Data parameters
DATA_DIR = "artportalen_images"  # Should contain 100 subfolders for the 100 species
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 100

# Training parameters
EPOCHS = 50
LEARNING_RATE = 1e-3

# Paths
CHECKPOINT_DIR = "checkpoints/"
LOG_DIR = "logs/"
MODEL_SAVE_PATH = "saved_model/bird_classifier.keras"
LABELS_PATH = "saved_model/bird_labels.json"
YOLO_MODEL_PATH = "yolov8n.pt" # Pre-trained YOLOv8 on COCO
TEST_IMAGES_PATH = "test_images/"