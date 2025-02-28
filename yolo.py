import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Use a different model if needed

# Load MiDaS depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

# Preprocessing for MiDaS
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def detect_pedestrians(image):
    results = yolo_model(image)  # Run YOLO on the image
    pedestrians = []
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            bbox = box.xyxy[0].cpu().numpy().astype(int)  # Bounding box [x1, y1, x2, y2]
            
            if class_id == 0:  # COCO class 0 for 'person'
                pedestrians.append((bbox, confidence))
    
    return pedestrians

def estimate_depth(image):
    image_resized = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_resized).unsqueeze(0)
    
    with torch.no_grad():
        depth_map = midas(image_tensor)
    
    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))  # Resize to original size
    
    return depth_map

def get_pedestrian_depth(image, pedestrians):
    depth_map = estimate_depth(image)
    depth_results = []
    
    for bbox, confidence in pedestrians:
        x1, y1, x2, y2 = bbox
        pedestrian_depth = depth_map[y1:y2, x1:x2]
        avg_depth = np.mean(pedestrian_depth)  # Compute average depth
        depth_results.append((bbox, confidence, avg_depth))
    
    return depth_results

def process_images(folder_path):
    for filename in os.listdir(folder_path):
        if filename.startswith(('A', 'C')) and filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Could not read image: {filename}")
                continue
            
            pedestrians = detect_pedestrians(image)
            depth_results = get_pedestrian_depth(image, pedestrians)
            
            for bbox, confidence, avg_depth in depth_results:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Depth: {avg_depth:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Run the script
process_images("Dataset_Occluded_Pedestrian")
