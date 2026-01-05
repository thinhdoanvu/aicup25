from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("cfg/models/v8/yolov8x_DW_swin_FOCUS-3.yaml")
    model.train(data="../AICUP25/aortic_valve.yaml",
                imgsz=640,
                batch=8,
                epochs=200,
                )
