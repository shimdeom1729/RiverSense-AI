import cv2
import numpy as np
from ultralytics import YOLO

class RiverVisionEngine:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.prev_gray = None

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Object Detection
        results = self.model(frame, conf=0.25, verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = self.model.names[cls_id]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            detections.append({
                'class': name,
                'confidence': round(conf, 2),
                'bbox': [x1, y1, x2, y2],
                'centroid': (cx, cy),
                'pixel_area': (x2 - x1) * (y2 - y1)
            })

        # 2. Optical Flow (Flow Speed)
        flow_vector = (0.45, 0.12) # Default fallback m/s proxy
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            u = float(np.mean(flow[..., 0]))
            v = float(np.mean(flow[..., 1]))
            if not np.isnan(u) and not np.isnan(v):
                flow_vector = (u, v)

        self.prev_gray = gray
        return detections, flow_vector