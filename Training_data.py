from ultralytics import YOLO

# Load a YOLOv5 model
model = YOLO('yolov5su.pt')

# Train the model
model.train(
    data='dataset/data.yaml', # Path to the dataset configuration file
    epochs=50,                # Number of training epochs
    imgsz=640,                # Image size (adjust as needed)
    batch=16,                 # Batch size
    device=0                  # Use 0 for GPU or -1 for CPU
)
    