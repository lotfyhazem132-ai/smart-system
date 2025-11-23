import cv2
import cv2.aruco as aruco

# --- DroidCam URL ---
droidcam_url = "http://192.168.1.12:4747/video"  # Replace with your DroidCam IP
cap = cv2.VideoCapture(droidcam_url)

# --- ArUco dictionary ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# --- Marker IDs, roles, and links ---
ids_roles = {
    0: {"role": "Engineer", "link": "https://example.com/engineer"},
    1: {"role": "Worker", "link": "https://example.com/worker"},
    2: {"role": "Visitor", "link": "https://example.com/visitor"}
}

# --- Bounding box colors (BGR) ---
colors = {
    "Engineer": (255, 255, 255),  # White
    "Worker": (0, 255, 255),      # Yellow
    "Visitor": (0, 0, 255)        # Red
}

print("Starting ArUco detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to get frame from DroidCam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids_detected, rejected = aruco.detectMarkers(gray, aruco_dict)

    if ids_detected is not None:
        ids_detected = ids_detected.flatten()
        for i, marker_id in enumerate(ids_detected):
            info = ids_roles.get(marker_id)
            if info:
                role = info["role"]
                link = info["link"]
                color = colors.get(role, (0, 255, 0))

                # Draw bounding box around marker
                cv2.polylines(frame, [corners[i].astype(int)], True, color, 3)

                # Draw role label above marker
                corner_pos = tuple(corners[i][0][0].astype(int) - [0, 10])
                cv2.putText(frame, role, corner_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Print the link for this marker
                print(f"Detected {role}: {link}")

    cv2.imshow("DroidCam ArUco Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()