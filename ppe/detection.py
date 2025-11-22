import cv2
import torch
import numpy as np
import os

# Set working directory to the folder containing the model and video file
os.chdir(r"D:\Desktop\yolov5safetyhelmet")

# Ensure output directory exists
if not os.path.exists("output"):
    os.makedirs("output")

# Load the model with error handling
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'best final.pt', force_reload=True)
except Exception as e:
    print("Error loading the model:", e)
    exit()

# Load the video file
video_file = "http://192.168.1.205:4747/video" 
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define output settings
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = 'output.mp4'
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback

out = cv2.VideoWriter(output_file, fourcc, fps, (1020, 600))

# Define font for labeling
font = cv2.FONT_HERSHEY_SIMPLEX
frame_idx = 0

# Process each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Run YOLO detection
    results = model(frame)
    pred = results.pred[0].cpu().numpy()

    helmets = pred[pred[:, 5] == 0]
    vests = pred[pred[:, 5] == 1]
    workers = pred[pred[:, 5] == 2]

    # Draw bounding boxes for helmets
    for helmet in helmets:
        x1, y1, x2, y2 = map(int, helmet[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Helmet', (x1, y1 - 10), font, 0.5, (0, 255, 0), 2)

    # Draw bounding boxes for vests
    for vest in vests:
        x1, y1, x2, y2 = map(int, vest[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, 'Vest', (x1, y1 - 10), font, 0.5, (255, 0, 0), 2)

    # Check for workers without safety gear
    for worker_idx, worker in enumerate(workers):
        x1, y1, x2, y2 = map(int, worker[:4])
        has_helmet = False
        has_vest = False

        # Match helmets + vests to workers
        for helmet in helmets:
            if x1 <= helmet[2] and x2 >= helmet[0] and y1 <= helmet[3] and y2 >= helmet[1]:
                has_helmet = True
                break

        for vest in vests:
            if x1 <= vest[2] and x2 >= vest[0] and y1 <= vest[3] and y2 >= vest[1]:
                has_vest = True
                break

        # Save cropped worker if missing gear
        if not has_helmet or not has_vest:
            worker_img = frame[y1:y2, x1:x2]
            worker_type = (
                "No Helmet and Vest" if not has_helmet and not has_vest
                else "No Helmet" if not has_helmet
                else "No Vest"
            )
            cv2.imwrite(f'output/{frame_idx}_{worker_type}_worker_{worker_idx}.png', worker_img)
            cv2.putText(frame, worker_type, (x1, y1 - 25), font, 0.5, (0, 0, 255), 2)

    # Draw bounding box around workers
    for worker in workers:
        x1, y1, x2, y2 = map(int, worker[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 255), 2)
        cv2.putText(frame, 'Worker', (x1, y1 - 10), font, 0.5, (255, 0, 255), 2)

    out.write(frame)
    cv2.imshow('frame', frame)
    frame_idx += 1

    # Stop on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing completed successfully. Output saved as", output_file)
