import PIL.Image as Image
from ultralytics import YOLO
import cv2
'''
model = YOLO('yolov8n.pt')
results = model('.../imagePath', classes=[14], conf=0.1) #can change conf to show more / less birds depending on accuracy
for r in results:
    im_array = r.plot(conf=False, labels=True)

    im = Image.fromarray(im_array[..., ::-1])
    im.show()
'''
def train_model():
    # Load the model
    model = YOLO('yolo11n-cls.pt')

    # Run the training

    model.train(
        data='artportalen_images',
        epochs=50,
        imgsz=224,
        device=0,
        workers=8,
        patience=10, #for early stopping
        verbose=False
    )

if __name__ == '__main__':
    '''train_model()'''
# 1. Load models
detector = YOLO('yolov8n.pt')
classifier = YOLO(r'D:\Python\flask\runs\classify\train-6\weights\best.pt')

# 2. Run detector on the test image
image_path = "domherre.png"
results = detector(image_path)

# Load the image with OpenCV for drawing
img = cv2.imread(image_path)

for r in results:
    for box in r.boxes:
        # Check if detected object is a bird (Class 14)
        if int(box.cls) == 14:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Create the crop for the classifier
            bird_crop = r.orig_img[y1:y2, x1:x2]

            # 3. Classify the crop
            species_results = classifier(bird_crop)

            # Get the top prediction name and confidence
            top_idx = species_results[0].probs.top1
            species_name = species_results[0].names[top_idx]
            confidence = species_results[0].probs.top1conf.item()

            print(f"Top Result: {species_name} ({confidence:.2%})")

            # 4. Draw bounding box and label on the image
            # Draw the box (Green)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw the label background
            label = f"{species_name} {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 5. Show the final result
cv2.imshow("Bird Classification Test", img)
cv2.waitKey(0)  # Press any key to close the window
cv2.destroyAllWindows()