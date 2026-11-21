
import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread("car.jpg", 0)
img2 = cv2.imread("rhishi.jpg", 0)

if img1 is None or img2 is None:
    raise FileNotFoundError("Images not found")

# SIFT Feature Matching
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

matcher = cv2.BFMatcher()
matches = matcher.knnMatch(des1, des2, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

match_img = cv2.drawMatches(img1, kp1, img2, kp2,
                            good[:50], None, flags=2)
cv2.imwrite("lab8a_output.jpg", match_img)

# Object Localization using Homography
if len(good) >= 4:
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    h, w = img1.shape
    corners = np.float32([[0,0], [0,h-1], [w-1,h-1],
                          [w-1,0]]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(corners, H)
    scene = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    localized = cv2.polylines(scene, [np.int32(projected)],
                              True, (0,255,0), 3)
    cv2.imwrite("lab8b_output.jpg", localized)
else:
    print("Not enough good matches for localization.")

plt.imshow(match_img, cmap="gray")
plt.title("SIFT Feature Matching")
plt.axis("off")
plt.show()
