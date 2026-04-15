from ultralytics import YOLO

model_yaml_path = '/home/qiqi/yolov8_obb/ultralytics/cfg/models/v8/X-RAY.yaml'
# 数据集配置文件
data_yaml_path = '/home/qiqi/yolov8_obb/ultralytics/cfg/datasets/X-RAY.yaml'
# 预训练模型
pre_model_name = '/home/qiqi/yolov8_obb/weights/yolov8n.pt'


def main():
    model = YOLO(model_yaml_path).load(pre_model_name)  # build from YAML and transfer weights

    model.train(data=data_yaml_path,
                epochs=200,
                imgsz=640,
                batch=4,
                workers=4,
                device=0,
                name="X-RAY")


if __name__ == '__main__':
    main()

# yolo obb train data=data/hat.yaml model=yolov8s-obb.pt epochs=200 imgsz=640 device=0