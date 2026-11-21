# LAB 4: Fourier Transform and Frequency-Domain Filtering

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale
img = cv2.imread("rhishi.jpg", cv2.IMREAD_GRAYSCALE)

if img is None:
    raise FileNotFoundError(
        "Place lab4.jpg in the same folder as this Python file."
    )


# ---------------------------------------------------------
# Part (a): Fourier Transform and Magnitude Spectrum
# ---------------------------------------------------------

# Apply 2D Fourier Transform
fourier = np.fft.fft2(img)

# Shift the zero frequency component to the center
fourier_shifted = np.fft.fftshift(fourier)

# Calculate magnitude spectrum
magnitude_spectrum = 20 * np.log1p(
    np.abs(fourier_shifted)
)


# ---------------------------------------------------------
# Calculate Distance from Center
# ---------------------------------------------------------

rows, cols = img.shape

center_row = rows // 2
center_col = cols // 2

# Cut-off frequency
D0 = 40

# Generate coordinate grids
y, x = np.ogrid[:rows, :cols]

# Calculate Euclidean distance from center
distance = np.sqrt(
    (x - center_col) ** 2 +
    (y - center_row) ** 2
)


# ---------------------------------------------------------
# Part (b): Low-Pass and High-Pass Masks
# ---------------------------------------------------------

# Create low-pass mask
low_pass_mask = np.zeros(
    (rows, cols),
    dtype=np.float32
)

# Keep frequencies inside D0
low_pass_mask[distance <= D0] = 1

# Create high-pass mask
high_pass_mask = 1 - low_pass_mask


# Apply frequency masks
low_pass_frequency = (
    fourier_shifted * low_pass_mask
)

high_pass_frequency = (
    fourier_shifted * high_pass_mask
)


# ---------------------------------------------------------
# Inverse Fourier Transform - Low-Pass Filter
# ---------------------------------------------------------

low_pass_inverse = np.fft.ifft2(
    np.fft.ifftshift(low_pass_frequency)
)

low_pass_output = np.abs(
    low_pass_inverse
)


# ---------------------------------------------------------
# Inverse Fourier Transform - High-Pass Filter
# ---------------------------------------------------------

high_pass_inverse = np.fft.ifft2(
    np.fft.ifftshift(high_pass_frequency)
)

high_pass_output = np.abs(
    high_pass_inverse
)


# ---------------------------------------------------------
# Normalize Outputs to 0-255
# ---------------------------------------------------------

low_pass_output = cv2.normalize(
    low_pass_output,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

high_pass_output = cv2.normalize(
    high_pass_output,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

spectrum_output = cv2.normalize(
    magnitude_spectrum,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)


# ---------------------------------------------------------
# Save Results
# ---------------------------------------------------------

cv2.imwrite(
    "lab4a_spectrum_output.jpg",
    spectrum_output
)

cv2.imwrite(
    "lab4b_lowpass_output.jpg",
    low_pass_output
)

cv2.imwrite(
    "lab4b_highpass_output.jpg",
    high_pass_output
)


# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------

images = [
    img,
    magnitude_spectrum,
    low_pass_mask,
    low_pass_output,
    high_pass_mask,
    high_pass_output
]

titles = [
    "Original Image",
    "Magnitude Spectrum",
    "Low-Pass Mask",
    "Low-Pass Output",
    "High-Pass Mask",
    "High-Pass Output"
]

plt.figure(figsize=(14, 8))

for i, image in enumerate(images):

    plt.subplot(2, 3, i + 1)

    plt.imshow(
        image,
        cmap="gray"
    )

    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()