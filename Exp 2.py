import cv2
import matplotlib.pyplot as plt

# Read the input image
img = cv2.imread("rhishi.jpg")

# Check whether image is loaded successfully
if img is None:
    raise FileNotFoundError(
        "Place lab2.jpg in the same folder as this Python file."
    )

# ---------------------------------------------------------
# Part (a): Brightness and Contrast Adjustment
# ---------------------------------------------------------

alpha = 1.5   # Contrast control
beta = 30     # Brightness control

# Adjust brightness and contrast
contrast_img = cv2.convertScaleAbs(
    img,
    alpha=alpha,
    beta=beta
)

# Save the contrast-adjusted image
cv2.imwrite("lab2a_output.jpg", contrast_img)


# ---------------------------------------------------------
# Part (b): Histogram Equalization
# ---------------------------------------------------------

# Convert original image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization
equalized = cv2.equalizeHist(gray)

# Save the equalized image
cv2.imwrite("lab2b_output.jpg", equalized)


# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

# Contrast Adjusted Image
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB))
plt.title("Contrast Adjusted")
plt.axis("off")

# Original Histogram
plt.subplot(2, 2, 3)
plt.hist(gray.ravel(), 256, [0, 256])
plt.title("Original Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# Histogram Equalized Image
plt.subplot(2, 2, 4)
plt.imshow(equalized, cmap="gray")
plt.title("Histogram Equalized")
plt.axis("off")

plt.tight_layout()
plt.show()