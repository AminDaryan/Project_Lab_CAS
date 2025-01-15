from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained YOLOv11 model
model = YOLO('runs/train/weights/best.pt')
model.to('cuda')

# Load an image
image_path = 'dataset/image1.png'
image = cv2.imread(image_path)

# Perform object detection
results = model(image)

# Render the results on the image
annotated_image = np.squeeze(results[0].plot())

# Display the image with detections
cv2.imshow('Detection Results', annotated_image)

# Save the annotated image to disk
output_path = 'output_image.jpg'
cv2.imwrite(output_path, annotated_image)
print(f"Detection results saved to {output_path}")

# Wait for a key press and close the display window
cv2.waitKey(0)
cv2.destroyAllWindows()