import os
import sys

# Import the training function and inference class from our other files
#from train import train
#from inference import BirdDetectionApp
from config import MODEL_SAVE_PATH, YOLO_MODEL_PATH, LABELS_PATH


def run_training():
    print("\n" + "=" * 40)
    print("   Initiating Training Pipeline")
    print("=" * 40)
    # Call the train function from train.py

    from train import train
    train()


def run_inference():
    print("\n" + "=" * 40)
    print("   Initiating Inference Pipeline")
    print("=" * 40)

    # Ensure the classification model has actually been trained before trying to use it
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Classification model not found at {MODEL_SAVE_PATH}.")
        print("Please run the training pipeline (Option 1) first to generate the model.")
        return

    # Prompt the user for the image they want to test
    image_path = input("\nEnter the file path to the test image (e.g., test_bird.jpg): ").strip()

    # Strip quotes just in case the user dragged and dropped the file into the terminal
    image_path = image_path.replace('"', '').replace("'", "")

    if not os.path.exists(image_path):
        print(f"Error: Could not find the image at '{image_path}'. Please check the path and try again.")
        return

    print(f"\nLoading models and processing '{image_path}'...")
    try:
        print('before')
        from inference import BirdDetectionApp
        print('after')

        app = BirdDetectionApp(MODEL_SAVE_PATH, YOLO_MODEL_PATH, LABELS_PATH)
        app.process_image(image_path)
    except Exception as e:
        print(f"An error occurred during inference: {e}")


def main():
    while True:
        print("\n" + "*" * 45)
        print("* Swedish Bird Classifier & Detector App  *")
        print("*" * 45)
        print("1. Train the classification model")
        print("2. Test model on a single image")
        print("3. Exit")

        choice = input("\nEnter your choice (1, 2, or 3): ").strip()

        if choice == '1':
            run_training()
        elif choice == '2':
            run_inference()
        elif choice == '3':
            print("Exiting application. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid input. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    # Ensure necessary packages are installed
    try:
        import tensorflow as tf
        import ultralytics
        import cv2
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages using: pip install tensorflow ultralytics opencv-python")
        sys.exit(1)

    import tensorflow as tf

    # 1. Get a list of all available physical GPUs
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # 2. Turn on runtime memory allocation (allow_growth) for the first GPU
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print("GPU memory growth enabled!")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)



    main()