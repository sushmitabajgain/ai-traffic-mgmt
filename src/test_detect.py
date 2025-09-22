from ultralytics import YOLO
import cv2

# VIDEO = "../data/sample.mp4"
# Change to 0 for webcam
VIDEO = 0

def main():
    model = YOLO("yolov8n.pt")  # auto-downloads on first run
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("Cannot open video/camera"); return

    while True:
        ok, frame = cap.read()
        if not ok: break

        # Run inference (Ultralytics expects RGB; it handles conversion internally)
        results = model.predict(source=frame, imgsz=640, conf=0.35, iou=0.45, verbose=False)
        annotated = results[0].plot()  # draw boxes on a copy

        cv2.imshow("YOLOv8 quick test", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
