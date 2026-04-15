import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/qiqi/yolov8_obb/runs/detect/X-RAY/weights/best.pt')
    model.predict(source='/home/qiqi/yolov8_obb/datasets/X-RAY/images/test/00052.jpg',
                  project='runs/detect',
                  name='exp',
                  save=True,
                  #show_conf=False
                  # visualize=True # visualize model features maps
                  )