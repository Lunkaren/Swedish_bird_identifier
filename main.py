import PIL.Image as Image
from ultralytics import YOLO


model = YOLO('yolov8n.pt')
results = model('.../imagePath', classes=[14], conf=0.1) #can change conf to show more / less birds depending on accuracy
for r in results:
    im_array = r.plot(conf=False, labels=True)

    im = Image.fromarray(im_array[..., ::-1])
    im.show()
