import numpy as np
import cv2

def apply_gaussian_kernel(image, sigma=0.6):

    # Compute kernel size: 6 * sigma rounded to the nearest odd integer
    kernel_size = int(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure kernel size is odd

    gaussian_kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2

    # Create the Gaussian kernel
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            gaussian_kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    # Normalize the kernel
    gaussian_kernel /= (2 * np.pi * sigma**2)
    gaussian_kernel /= np.sum(gaussian_kernel)

    # Apply the Gaussian kernel to the image
    padded_image = np.pad(image, center, mode='reflect')
    smoothed_image = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            smoothed_image[i, j] = np.sum(region * gaussian_kernel)

    return smoothed_image

def compute_magnitude(image, sigma=0.6):
#Compute the gradient magnitude of an image using Gaussian smoothing and finite difference.

    smoothed_image = apply_gaussian_kernel(image, sigma)
    gradient_x = np.zeros_like(smoothed_image)
    gradient_y = np.zeros_like(smoothed_image)

    # Compute gradients using finite difference approximations
    for i in range(1, smoothed_image.shape[0] - 1):
        for j in range(1, smoothed_image.shape[1] - 1):
            # Approximate gradient in the x direction
            gradient_x[i, j] = (smoothed_image[i, j + 1] - smoothed_image[i, j - 1]) / 2

            # Approximate gradient in the y direction
            gradient_y[i, j] = (smoothed_image[i + 1, j] - smoothed_image[i - 1, j]) / 2

    # Compute the magnitude of the gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)

    return magnitude

def WaterShed(img):

    label = -np.ones_like(img, dtype=np.int32)  # Set label[p] = -1 for all pixels (unlabeled)
    global_label = 0

    # Precompute the array of pixel lists for each gray level g
    G = 256  # Assuming an 8-bit image
    gray_levels = [[] for _ in range(G)]
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            gray_levels[img[x, y]].append((x, y))

    # Flood topological surface one gray level at a time
    for g in range(G):
        temp_label = label.copy()  # Copy of the current label state
        frontier = []  # Initialize as an empty list

        # Grow existing catchment basins by one pixel, creating initial frontier
        for (x, y) in gray_levels[g]:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check 4 neighbors
                nx, ny = x + dx, y + dy
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                    if label[nx, ny] >= 0:  # If neighbor is labeled
                        label[x, y] = label[nx, ny]
                        frontier.append((x, y))
                        break

        # Continue to grow basins by expanding the frontier
        while frontier:
            px, py = frontier.pop(0)  # Pop the first element (FIFO behavior)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check 4 neighbors
                nx, ny = px + dx, py + dy
                if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                    if img[nx, ny] == g and label[nx, ny] == -1:  # If unlabeled pixel with current gray level
                        label[nx, ny] = label[px, py]
                        frontier.append((nx, ny))

        # Create new catchment basins
        for (x, y) in gray_levels[g]:
            if label[x, y] == -1:  # Still unlabeled
                global_label += 1
                label[x, y] = global_label
                frontier.append((x, y))

                # Flood-fill the region with the new global label
                while frontier:
                    px, py = frontier.pop(0)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Check 4 neighbors
                        nx, ny = px + dx, py + dy
                        if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]:
                            if img[nx, ny] == g and label[nx, ny] == -1:
                                label[nx, ny] = global_label
                                frontier.append((nx, ny))

    return label



