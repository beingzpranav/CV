import cv2
import matplotlib.pyplot as plt

# Read the input image
img = cv2.imread("rhishi.jpg")

if img is None:
    raise FileNotFoundError(
        "Place car.jpg in the same folder as this Python file."
    )

# Save original image
cv2.imwrite("lab1_output.jpg", img)

# Convert BGR to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert image to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Generate negative/complement image
negative = 255 - img

# Save processed images
cv2.imwrite("lab1a_gray_output.jpg", gray)
cv2.imwrite("lab1b_complement_output.jpg", negative)

# Display all images
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.imshow(rgb)
plt.title("RGB Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
plt.title("HSV View")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.title("Complement")
plt.axis("off")

plt.tight_layout()
plt.show()