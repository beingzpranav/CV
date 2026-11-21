
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

img = cv2.imread("rhishi.jpg")
if img is None:
    raise FileNotFoundError("Image not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# SIFT
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
sift_img = cv2.drawKeypoints(img, keypoints, None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# HOG
features, hog_image = hog(gray, orientations=9,
    pixels_per_cell=(8, 8), cells_per_block=(2, 2),
    visualize=True, block_norm="L2-Hys")

hog_vis = exposure.rescale_intensity(hog_image,
    in_range=(0, hog_image.max()))
hog_vis = (255 * hog_vis).astype(np.uint8)

cv2.imwrite("lab6a_output.jpg", sift_img)
cv2.imwrite("lab6b_output.jpg", hog_vis)

print("SIFT keypoints:", len(keypoints))
print("HOG feature length:", len(features))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(sift_img, cv2.COLOR_BGR2RGB))
plt.title("SIFT")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hog_vis, cmap="gray")
plt.title("HOG")
plt.axis("off")

plt.tight_layout()
plt.show()
