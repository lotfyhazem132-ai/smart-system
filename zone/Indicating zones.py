import cv2
import numpy as np
import json
import os

# --------------------------
# Camera setup
cap = cv2.VideoCapture("http://192.168.1.9:4747/video")
key = cv2.waitKey(1) & 0xFF

# --------------------------
# Zones storage
zones = []         # Load from file if it exists
current_zone = []  # currently drawing zone

# Load zones from file at startup
if os.path.exists("zones.json"):
    with open("zones.json", "r") as f:
        zones = json.load(f)
    print(f"Loaded {len(zones)} zones from zones.json")
else:
    print("No zones.json found. Starting fresh.")

# --------------------------
# Mouse callback for drawing zones
def draw_zone(event, x, y, flags, param):
    global current_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        current_zone.append([x, y])
        print(f"Point added: {[x, y]}")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_zone) >= 3:  # polygon needs at least 3 points
            zone_name = input("Enter zone name: ").strip()
            if not zone_name:
                zone_name = f"Zone_{len(zones) + 1}"
            zones.append({"name": zone_name, "points": current_zone.copy()})
            print(f"Zone '{zone_name}' saved with {len(current_zone)} points")
            current_zone.clear()
        else:
            print("Zone needs at least 3 points. Right-click to save.")

cv2.namedWindow("Zone Drawing")
cv2.setMouseCallback("Zone Drawing", draw_zone)

# --------------------------
# Function to check if a point is inside a polygon
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0

# --------------------------
# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
    # Draw current polygon being drawn (red)
    if len(current_zone) > 0:
        pts = current_zone + [current_zone[0]]
        for i in range(len(current_zone)):
            cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (0,0,255), 2)
        cv2.putText(frame, "Drawing zone... (Right-click to finish)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # Draw saved zones (green)
    for zone_obj in zones:
        zone = zone_obj["points"]
        pts = zone + [zone[0]]
        for i in range(len(zone)):
            cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), (0,255,0), 2)
        # Draw zone name at first point
        cv2.putText(frame, zone_obj["name"], tuple(zone[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # --------------------------
    # Simulated human detection (replace later with AI)
    # For testing, use a moving rectangle
    human_bbox = (200, 100, 260, 220)  # xmin, ymin, xmax, ymax
    cx = (human_bbox[0] + human_bbox[2]) // 2
    cy = (human_bbox[1] + human_bbox[3]) // 2

    # Draw human rectangle and center
    cv2.rectangle(frame, (human_bbox[0], human_bbox[1]), (human_bbox[2], human_bbox[3]), (255,0,0), 2)
    cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

    # Check which zone human is in
    for zone_obj in zones:
        zone = zone_obj["points"]
        if point_in_polygon((cx, cy), zone):
            cv2.putText(frame, f"Human in {zone_obj['name']}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Display frame
    cv2.imshow("Zone Drawing", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # Quit
        break
    elif key == ord('s'):  # Save zones
        with open("zones.json","w") as f:
            json.dump(zones, f, indent=2)
        print("Zones saved to zones.json")
    elif key == ord('c'):  # Clear all zones
        zones.clear()
        print("All zones cleared")

cap.release()
cv2.destroyAllWindows()
