
from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolov8n.pt")

# Object detection on image
results = model("rhishi.jpg")
annotated = results[0].plot()
cv2.imwrite("lab9a_output.jpg", annotated)

# Object detection on video
cap = cv2.VideoCapture("lab9_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model(frame, verbose=False)
    frame_out = r[0].plot()

    if out is None:
        h, w = frame_out.shape[:2]
        out = cv2.VideoWriter(
            "lab9b_output.mp4", fourcc, 20, (w, h))

    out.write(frame_out)

cap.release()

if out is not None:
    out.release()

print("YOLO detection completed.")
