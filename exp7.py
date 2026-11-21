
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

img = cv2.imread("rhishi.jpg")
if img is None:
    raise FileNotFoundError("Image not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Global Thresholding
_, global_th = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive Thresholding
adaptive_th = cv2.adaptiveThreshold(gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# K-Means Clustering
pixels = img.reshape((-1, 3)).astype(np.float32)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(pixels)
centers = np.uint8(kmeans.cluster_centers_)
segmented = centers[labels].reshape(img.shape)

# Display Results
images = [img, global_th, adaptive_th, segmented]
titles = ["Original", "Global", "Adaptive", "K-Means"]

plt.figure(figsize=(12, 4))
for i, (image, title) in enumerate(zip(images, titles), 1):
    plt.subplot(1, 4, i)
    plt.imshow(image if image.ndim == 2 else
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
