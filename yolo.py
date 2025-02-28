import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import cv2
import numpy as np
import torch
import base64
import json
from ultralytics import YOLO
from google.cloud import pubsub_v1

# Load Models
yolo_model = YOLO("yolov8n.pt")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()

def detect_pedestrians(image):
    results = yolo_model(image)
    pedestrians = []
    for result in results:
        for box in result.boxes:
            if int(box.cls.item()) == 0:  # COCO class 0 = 'person'
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                pedestrians.append(bbox)
    return pedestrians

def estimate_depth(image):
    transform = torch.nn.Sequential(
        torch.nn.Resize((384, 384)),
        torch.nn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    image_resized = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.tensor(image_resized).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        depth_map = midas(transform(image_tensor)).squeeze().cpu().numpy()
    return depth_map

class ProcessImage(beam.DoFn):
    def process(self, element):
        message = json.loads(element.decode("utf-8"))
        image_data = base64.b64decode(message["image"])
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        
        pedestrians = detect_pedestrians(image)
        depth_map = estimate_depth(image)
        
        results = []
        for bbox in pedestrians:
            x1, y1, x2, y2 = bbox
            avg_depth = np.mean(depth_map[y1:y2, x1:x2])
            results.append({"bbox": bbox.tolist(), "depth": avg_depth})
        
        yield json.dumps(results)

def run_pipeline(input_topic, output_topic, project_id):
    options = PipelineOptions(
        streaming=True,
        project=project_id,
        runner="DataflowRunner",
        region="us-central1"
    )
    
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | "Read from Pub/Sub" >> beam.io.ReadFromPubSub(topic=input_topic)
            | "Process Image" >> beam.ParDo(ProcessImage())
            | "Write to Pub/Sub" >> beam.io.WriteToPubSub(topic=output_topic)
        )

# Run locally for testing
if __name__ == "__main__":
    run_pipeline(
        "projects/YOUR_PROJECT_ID/topics/input-topic",
        "projects/YOUR_PROJECT_ID/topics/output-topic",
        "YOUR_PROJECT_ID"
    )
