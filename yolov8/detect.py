from ultralytics import YOLO
model = YOLO('/home/qiqi/yolov8_obb/runs/obb/ta4/weights/best.pt')
results = model(
    'datasets/ta-well/images/test/DJI_20241022164637_0016_Z.JPG',
    save=True)