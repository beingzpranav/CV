
from ultralytics import YOLO
import cv2, glob, os, time

# Load YOLO model
model = YOLO("yolov8n.pt")
os.makedirs("lab10_outputs", exist_ok=True)

# Load dataset images
image_paths = glob.glob("dataset_images/*.jpg")
if not image_paths:
    raise FileNotFoundError("No JPG images found")

# Object detection and time measurement
inference_times = []

for path in image_paths:
    start = time.time()
    results = model(path, verbose=False)
    inference_times.append(time.time() - start)

    annotated = results[0].plot()
    cv2.imwrite(os.path.join("lab10_outputs",
                os.path.basename(path)), annotated)

# Save output image
img = cv2.imread(os.path.join(
    "lab10_outputs", os.path.basename(image_paths[0])))
cv2.imwrite("lab10a_output.jpg", img)

# Calculate average inference time
avg_time = sum(inference_times) / len(inference_times)

with open("lab10b_output.txt", "w") as f:
    f.write(f"Total Images: {len(image_paths)}\n")
    f.write(f"Average Inference Time: {avg_time:.4f} seconds/image\n")

print("Total Images:", len(image_paths))
print("Average Inference Time:", round(avg_time, 4), "seconds/image")
