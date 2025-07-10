from ultralytics import YOLO


if __name__ == '__main__':

    model = YOLO('yolov8n.pt')

    results = model.train(
        data='data.yaml',
        epochs=50,
        lr0=0.005,
        lrf=0.01,
        optimizer='AdamW',
        batch=32,
        imgsz=640,
        device='cuda'
    )