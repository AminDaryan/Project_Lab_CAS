import os

# Clear terminal before running anything
os.system('cls' if os.name == 'nt' else 'clear')

from ultralytics import YOLO
import torch

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)  # Ensure proper multiprocessing handling

    # Load a model
    model = YOLO("yolov8n.pt")  # Load pre-trained model

    # Use the model
    results = model.train(
        data=os.path.abspath(os.path.join("datasets/Target_Cube_Object_Detection_2", "data.yaml")),
        epochs=20,
        workers=0  # Disable multiprocessing to avoid Windows issues
    )
