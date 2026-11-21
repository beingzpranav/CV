# LAB 3: Spatial-domain smoothing and sharpening

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
img = cv2.imread("rhishi.jpg")

if img is None:
    raise FileNotFoundError(
        "Place lab3.jpg in the same folder as this Python file."
    )


# ---------------------------------------------------------
# Part (a): Image Smoothing
# ---------------------------------------------------------

# Average filter
average_img = cv2.blur(img, (5, 5))

# Gaussian filter
gaussian_img = cv2.GaussianBlur(
    img,
    (5, 5),
    sigmaX=1.0
)

# Median filter
median_img = cv2.medianBlur(img, 5)

# Save smoothing results
cv2.imwrite(
    "lab3a_average_output.jpg",
    average_img
)

cv2.imwrite(
    "lab3a_gaussian_output.jpg",
    gaussian_img
)

cv2.imwrite(
    "lab3a_median_output.jpg",
    median_img
)


# ---------------------------------------------------------
# Part (b): Laplacian Sharpening
# ---------------------------------------------------------

# Convert image to floating-point format
img_float = img.astype(np.float32)

# Define Laplacian kernel
laplacian_kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)

# Apply Laplacian filter
laplacian_response = cv2.filter2D(
    img_float,
    cv2.CV_32F,
    laplacian_kernel
)

# Sharpen the image
laplacian_sharp = img_float - laplacian_response

# Keep pixel values between 0 and 255
laplacian_sharp = np.clip(
    laplacian_sharp,
    0,
    255
).astype(np.uint8)


# ---------------------------------------------------------
# Part (c): High-Pass Sharpening
# ---------------------------------------------------------

# Define high-pass kernel
high_pass_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

# Apply high-pass filter
high_pass_response = cv2.filter2D(
    img_float,
    cv2.CV_32F,
    high_pass_kernel
)

# Sharpening strength
alpha = 0.5

# Add high-frequency details to original image
high_pass_sharp = (
    img_float + alpha * high_pass_response
)

# Keep pixel values between 0 and 255
high_pass_sharp = np.clip(
    high_pass_sharp,
    0,
    255
).astype(np.uint8)


# Save sharpening results
cv2.imwrite(
    "lab3b_laplacian_output.jpg",
    laplacian_sharp
)

cv2.imwrite(
    "lab3b_highpass_output.jpg",
    high_pass_sharp
)


# ---------------------------------------------------------
# Display All Results
# ---------------------------------------------------------

images = [
    img,
    average_img,
    gaussian_img,
    median_img,
    laplacian_sharp,
    high_pass_sharp
]

titles = [
    "Original",
    "Average Filter",
    "Gaussian Filter",
    "Median Filter",
    "Laplacian Sharpening",
    "High-Pass Sharpening"
]

plt.figure(figsize=(14, 8))

for i, image in enumerate(images):

    plt.subplot(2, 3, i + 1)

    # Convert BGR to RGB for Matplotlib
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    plt.imshow(rgb)
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()