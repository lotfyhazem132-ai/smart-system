import cv2
from ultralytics import YOLO
import numpy as np
import os


class PPEDetector:
    def __init__(self, model_path = "D:\\Desktop\\yolov11safetyhelmet\\best (4).pt"):
        print(f"Loading YOLOv11 model from: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = YOLO(model_path)
        self.class_names = self.model.names

        # Detect ONLY these violations
        self.violation_classes = ['NO-Hardhat', 'NO-Safety-Vest']

        print(f"Monitoring violations: {self.violation_classes}")

    def is_violation(self, class_name):
        return class_name in self.violation_classes

    def detect_mobile_camera(self, camera_url):
        """Detect using mobile IP camera stream"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception(f"Cannot open mobile camera stream: {camera_url}")

        print("✔ Connected to mobile camera")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Lost connection to camera.")
                break

            results = self.model(frame, conf=0.4, iou=0.45)

            annotated_frame = frame.copy()

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.class_names[cls_id]

                   
                    if class_name in self.violation_classes:
                        # Red box for violation
                        color = (0, 0, 255)
                    else:
                        continue  # Ignore masks completely

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated_frame,
                        f"{class_name} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2
                    )

            cv2.imshow("PPE Detector (Mobile Camera)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = PPEDetector("D:\\Desktop\\yolov11safetyhelmet\\best (4).pt")


    # Change this to your mobile IP camera URL:
    mobile_camera_url = "http://192.168.1.145:4747/mjpegfeed"

    detector.detect_mobile_camera(mobile_camera_url)
