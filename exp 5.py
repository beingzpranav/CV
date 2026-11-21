# LAB 5: Edge Detection and Morphological Operations

import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("rhishi.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(
        "Place lab5.jpg in the same folder as this Python file."
    )


# ---------------------------------------------------------
# Part (a): Edge Detection
# ---------------------------------------------------------

# Sobel X direction
sobelx = cv2.Sobel(
    img,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

# Sobel Y direction
sobely = cv2.Sobel(
    img,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

# Combine Sobel X and Y
sobel = cv2.magnitude(
    sobelx.astype(np.float32),
    sobely.astype(np.float32)
)

# Normalize Sobel result
sobel = cv2.normalize(
    sobel,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


# Canny Edge Detection
canny = cv2.Canny(
    img,
    100,
    200
)


# ---------------------------------------------------------
# Prewitt Operator
# ---------------------------------------------------------

prewitt_x_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

prewitt_y_kernel = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
], dtype=np.float32)

prewitt_x = cv2.filter2D(
    img,
    cv2.CV_32F,
    prewitt_x_kernel
)

prewitt_y = cv2.filter2D(
    img,
    cv2.CV_32F,
    prewitt_y_kernel
)

# Combine Prewitt X and Y
prewitt = cv2.magnitude(
    prewitt_x,
    prewitt_y
)

# Normalize Prewitt result
prewitt = cv2.normalize(
    prewitt,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


# ---------------------------------------------------------
# Part (b): Morphological Operations
# ---------------------------------------------------------

# Convert image to binary
_, binary = cv2.threshold(
    img,
    127,
    255,
    cv2.THRESH_BINARY
)

# Create 5x5 kernel
kernel = np.ones(
    (5, 5),
    np.uint8
)


# Erosion
erosion = cv2.erode(
    binary,
    kernel,
    iterations=1
)


# Dilation
dilation = cv2.dilate(
    binary,
    kernel,
    iterations=1
)


# Opening
opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)


# Closing
closing = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel
)


# ---------------------------------------------------------
# Save Results
# ---------------------------------------------------------

cv2.imwrite("lab5a_sobel_output.jpg", sobel)
cv2.imwrite("lab5a_canny_output.jpg", canny)
cv2.imwrite("lab5a_prewitt_output.jpg", prewitt)

cv2.imwrite("lab5b_erosion_output.jpg", erosion)
cv2.imwrite("lab5b_dilation_output.jpg", dilation)
cv2.imwrite("lab5b_opening_output.jpg", opening)
cv2.imwrite("lab5b_closing_output.jpg", closing)


# ---------------------------------------------------------
# Display All Results
# ---------------------------------------------------------

images = [
    img,
    sobel,
    canny,
    prewitt,
    erosion,
    dilation,
    opening,
    closing
]

titles = [
    "Original",
    "Sobel",
    "Canny",
    "Prewitt",
    "Erosion",
    "Dilation",
    "Opening",
    "Closing"
]

# Create 2 rows and 4 columns
plt.figure(figsize=(16, 8))

for i, image in enumerate(images):

    plt.subplot(2, 4, i + 1)

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()