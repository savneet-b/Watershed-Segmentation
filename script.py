import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from SKBlib import WaterShed, compute_magnitude

def main():
    # Create a Tkinter root window and withdraw it to avoid displaying the main window.
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog for the user to select an image file, restricting to specific image types.
    img_path = filedialog.askopenfilename(
        title="Select the Image",  # The title of the file dialog window.
        filetypes=[("Image files", "*.bmp *.jpg *.jpeg *.png *.pgm")]  # Accepted file formats.
    )

    # Check if no file was selected and exit if true.
    if not img_path:
        print("No file selected. Exiting...")
        return

    # Load the selected image in grayscale mode.
    original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Check if the image was loaded correctly; if not, display an error and exit.
    if original_img is None:
        print("Error: Could not load the image. Please select a valid image file.")
        return

    # Display the original grayscale image in a window.
    cv2.imshow('Original Image', original_img)

    # Step 1: Compute the gradient magnitude of the image
    magnitude = compute_magnitude(original_img, sigma=0.6)
    # Display the computed gradient magnitude.
    cv2.imshow('Normal Watershed: Magnitude', magnitude)

    # Step 2: Apply the watershed algorithm to the gradient magnitude image.
    label = WaterShed(magnitude)
    # Normalize the labels to a displayable range (0-255) and convert to an 8-bit image.
    label_display = (255 * label / np.max(label)).astype(np.uint8)
    # Display the labeled regions resulting from the watershed segmentation.
    cv2.imshow('Normal Watershed: Labels', label_display)

    # Step 3: Perform thresholding using to create a binary image.
    _, threshold_img = cv2.threshold(original_img, 115, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Display the thresholded image.
    cv2.imshow('Marker Watershed: Threshold', threshold_img)

    # Step 4: Calculate the chamfer distance transform of the binary thresholded image.
    distance = cv2.distanceTransform(threshold_img, cv2.DIST_L2, 5)
    # Normalize the distance transform to a displayable range (0-255).
    chamfer_display = (255 * distance / np.max(distance)).astype(np.uint8)
    # Display the chamfer distance image.
    cv2.imshow('Marker Watershed: Chamfer', chamfer_display)

    # Step 5: Apply the watershed algorithm to the distance-transformed image.
    watershed_chamfer = WaterShed(chamfer_display)
    # Normalize the watershed result and convert to an 8-bit image.
    watershed_chamfer_display = (255 * watershed_chamfer / np.max(watershed_chamfer)).astype(np.uint8)
    # Display the watershed result of the chamfer image.
    cv2.imshow('Marker Watershed: Watershed of Chamfer', watershed_chamfer_display)

    # Step 6: Perform edge detection on the watershed result using the Canny algorithm.
    edges = cv2.Canny(watershed_chamfer_display, 100, 200)
    # Display the edges detected in the watershed result.
    cv2.imshow('Marker Watershed: Edges separating objects', edges)

    # Step 7: Combine the thresholded image and edges to create a marker image.
    # This image highlights the edges of segmented objects.
    marker_image = cv2.bitwise_or(threshold_img, edges)
    # Display the combined marker image.
    cv2.imshow('Marker Watershed: Watershed Marker', marker_image)

    # Wait for any key press to close the displayed windows.
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
