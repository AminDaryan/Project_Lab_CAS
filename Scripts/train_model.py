import os

# Clear terminal before running anything
os.system('cls' if os.name == 'nt' else 'clear')

from ultralytics import YOLO
import torch

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)  # Ensure proper multiprocessing handling

    # Load a model
    model = YOLO("yolo11n.pt")  # load pre trained model

    # Use the model
    
    # Train the model
    train_results = model.train(
        data=os.path.abspath(os.path.join("datasets/Target_Cube_Object_Detection_3", "data.yaml")),  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )
    
    # Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    results = model("path/to/image.jpg")
    results[0].show()

    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model