from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

results = model.train(
    data="./license_plate_training_data/data.yaml",
    epochs=10,
    save_period=10,
    workers=4,
)
