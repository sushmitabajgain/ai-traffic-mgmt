from ultralytics import YOLO
import numpy as np

VEHICLE_WORDS = {"car","bus","truck","motorcycle","motorbike","bicycle","van"}

class UltralyticsPTDetector:
    def __init__(self, model_name="yolov8n.pt", conf=0.35, iou=0.45, imgsz=640, device=""):
        self.model = YOLO(model_name)
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))

    def _is_vehicle(self, cls_name):
        return any(k in cls_name.lower() for k in VEHICLE_WORDS)

    def detect(self, frame):
        res = self.model.predict(source=frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
        dets = []
        if len(res.boxes) == 0: return dets
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        for (x1,y1,x2,y2), cf, ci in zip(xyxy, confs, clss):
            cls = self.names.get(int(ci), str(ci))
            if not self._is_vehicle(cls): continue
            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
            cx, cy = x + w//2, y + h//2
            dets.append({"bbox": (x,y,w,h), "center": (cx,cy), "conf": float(cf), "class": cls})
        return dets
