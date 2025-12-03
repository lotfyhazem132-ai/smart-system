import cv2
from ultralytics import YOLO
import numpy as np
from zone import ZONES   # import your zone polygons
import time
import json
import os

# Load your YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("http://192.168.1.9:4747/video")
key = cv2.waitKey(1) & 0xFF

# Buzzer (placeholder)
def buzzer_on():
    print("BUZZER: HUMAN IN DANGER ZONE!!")

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

# Load zones from file if it exists
zones = []
if os.path.exists("zones.json"):
    with open("zones.json", "r") as f:
        zones = json.load(f)
    print(f"Loaded {len(zones)} zones from zones.json")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Now safe to resize
    frame = cv2.resize(frame, (1280, 720))

    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                center = (cx, cy)

                # Draw detected person center
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # Check static ZONES and act based on zone name
                detected_zones = []
                for name, polygon in ZONES.items():
                    if point_in_polygon(center, polygon["points"]):
                        detected_zones.append(name)
                        if name == "danger_zone":
                            buzzer_on()

                # Print all detected zones (will print both danger and orange if both)
                for z in detected_zones:
                    if z == "danger_zone":
                        print("Person in DANGER ZONE!")
                    elif z == "orange_zone":
                        print("Person in ORANGE ZONE!")

                # Check loaded zones from JSON (dynamic) and act by name
                for zone_obj in zones:
                    zname = zone_obj.get("name", "").lower()
                    zpoints = zone_obj.get("points", [])
                    if not zpoints:
                        continue
                    if point_in_polygon(center, zpoints):
                        if "danger" in zname:
                            buzzer_on()
                            print(f"Person detected in {zone_obj.get('name','<zone>')}")
                        elif "orange" in zname:
                            print(f"Person detected in orange zone ({zone_obj.get('name','')})")
                        else:
                            pass

    # draw static and dynamic zones (unchanged)
    for zone in ZONES.values():
        cv2.polylines(frame, [np.array(zone["points"], np.int32)], True, zone["color"], 2)

    for zone_obj in zones:
        if "points" in zone_obj and zone_obj["points"]:
            cv2.polylines(frame, [np.array(zone_obj["points"], np.int32)], True, (255, 255, 0), 2)
            cv2.putText(frame, zone_obj.get("name",""), tuple(zone_obj["points"][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    cv2.imshow("Detect", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit
        break
cap.release()
cv2.destroyAllWindows()
