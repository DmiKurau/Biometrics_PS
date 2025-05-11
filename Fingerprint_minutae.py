import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from enum import Enum

# Global variables to store analysis results
results_cache = {}
current_algo = None
comparison_mode = False

class ThinningAlgorithm(Enum):
    ZHANG_SUEN = "zhang-suen"
    GUO_HALL = "guo-hall"


def zhang_suen_thinning(image):
    """
    Implementation of Zhang-Suen thinning algorithm.
    Expects a binary image with foreground as 1 and background as 0.
    """
    # Make a copy of the image
    skeleton = image.copy()

    # Convert to binary format where foreground is 1 and background is 0
    if skeleton.max() > 1:
        skeleton = skeleton // 255

    changing = True

    # Loop until no more changes
    while changing:
        # First sub-iteration
        changing = False
        marked = []

        rows, cols = skeleton.shape

        # Pad the image to handle border pixels
        padded = np.pad(skeleton, ((1, 1), (1, 1)), 'constant')

        # First sub-iteration
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if padded[i, j] == 1:  # Foreground pixel
                    # Get 8 neighbors
                    P2 = padded[i - 1, j]
                    P3 = padded[i - 1, j + 1]
                    P4 = padded[i, j + 1]
                    P5 = padded[i + 1, j + 1]
                    P6 = padded[i + 1, j]
                    P7 = padded[i + 1, j - 1]
                    P8 = padded[i, j - 1]
                    P9 = padded[i - 1, j - 1]

                    # Calculate B(P1)
                    B_P1 = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

                    # Calculate A(P1) - number of 01 patterns
                    neighbors = [P2, P3, P4, P5, P6, P7, P8, P9, P2]  # P2 is repeated to check P9->P2
                    A_P1 = 0
                    for k in range(8):
                        if neighbors[k] == 0 and neighbors[k + 1] == 1:
                            A_P1 += 1

                    # Check all conditions
                    condition_a = 2 <= B_P1 <= 6
                    condition_b = A_P1 == 1
                    condition_c = P2 * P4 * P6 == 0
                    condition_d = P4 * P6 * P8 == 0

                    if condition_a and condition_b and condition_c and condition_d:
                        marked.append((i - 1, j - 1))  # Mark for deletion
                        changing = True

        # Delete marked pixels
        for point in marked:
            skeleton[point] = 0

        # Second sub-iteration
        marked = []

        # Update padded skeleton
        padded = np.pad(skeleton, ((1, 1), (1, 1)), 'constant')

        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if padded[i, j] == 1:  # Foreground pixel
                    # Get 8 neighbors
                    P2 = padded[i - 1, j]
                    P3 = padded[i - 1, j + 1]
                    P4 = padded[i, j + 1]
                    P5 = padded[i + 1, j + 1]
                    P6 = padded[i + 1, j]
                    P7 = padded[i + 1, j - 1]
                    P8 = padded[i, j - 1]
                    P9 = padded[i - 1, j - 1]

                    # Calculate B(P1)
                    B_P1 = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

                    # Calculate A(P1) - number of 01 patterns
                    neighbors = [P2, P3, P4, P5, P6, P7, P8, P9, P2]  # P2 is repeated to check P9->P2
                    A_P1 = 0
                    for k in range(8):
                        if neighbors[k] == 0 and neighbors[k + 1] == 1:
                            A_P1 += 1

                    # Check all conditions (modified c and d for second sub-iteration)
                    condition_a = 2 <= B_P1 <= 6
                    condition_b = A_P1 == 1
                    condition_c_prime = P2 * P4 * P8 == 0
                    condition_d_prime = P2 * P6 * P8 == 0

                    if condition_a and condition_b and condition_c_prime and condition_d_prime:
                        marked.append((i - 1, j - 1))  # Mark for deletion
                        changing = True

        # Delete marked pixels
        for point in marked:
            skeleton[point] = 0

    # Convert back to the original format
    if image.max() > 1:
        skeleton = skeleton * 255

    return skeleton.astype(np.uint8)


def guo_hall_thinning(image):
    """
    Implementation of Guo-Hall thinning algorithm.
    Expects a binary image with foreground as 1 and background as 0.
    """
    # Make a copy of the image
    skeleton = image.copy()

    # Convert to binary format where foreground is 1 and background is 0
    if skeleton.max() > 1:
        skeleton = skeleton // 255

    # Create a padded version for easy neighborhood operations
    rows, cols = skeleton.shape
    padded = np.pad(skeleton, ((1, 1), (1, 1)), 'constant')

    # Initialize previous image for change detection
    prev = np.zeros_like(padded)

    # Continue until no changes
    changed = True

    while changed:
        # Update previous image
        np.copyto(prev, padded)

        # Perform two sub-iterations
        for iter_type in [0, 1]:
            # Create marker matrix to store points to be deleted
            marker = np.zeros_like(padded)

            # Scan through the image
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):
                    if padded[i, j] == 1:  # If foreground pixel
                        # Get 8 neighbors in P2, P3, ..., P9 format
                        p2 = int(padded[i - 1, j])
                        p3 = int(padded[i - 1, j + 1])
                        p4 = int(padded[i, j + 1])
                        p5 = int(padded[i + 1, j + 1])
                        p6 = int(padded[i + 1, j])
                        p7 = int(padded[i + 1, j - 1])
                        p8 = int(padded[i, j - 1])
                        p9 = int(padded[i - 1, j - 1])

                        # Calculate Guo-Hall criteria
                        # C is the number of distinct 0-1 patterns
                        C = ((not p2) and (p3 or p4)) + \
                            ((not p4) and (p5 or p6)) + \
                            ((not p6) and (p7 or p8)) + \
                            ((not p8) and (p9 or p2))

                        # N1 and N2 are the number of 1-valued neighbors
                        N1 = (p9 or p2) + (p3 or p4) + (p5 or p6) + (p7 or p8)
                        N2 = (p2 or p3) + (p4 or p5) + (p6 or p7) + (p8 or p9)

                        # Take minimum of N1 and N2
                        N = min(N1, N2)

                        # Check condition based on iteration type
                        if iter_type == 0:
                            m = ((p6 or p7 or (not p9)) and p8)
                        else:
                            m = ((p2 or p3 or (not p5)) and p4)

                        # If all conditions are met, mark pixel for deletion
                        if C == 1 and (2 <= N <= 3) and m == 0:
                            marker[i, j] = 1

            # Remove all marked pixels
            padded[marker == 1] = 0

        # Check if there were any changes
        changed = not np.array_equal(padded, prev)

    # Extract the result (without padding)
    skeleton = padded[1:-1, 1:-1]

    # Convert back to original format if needed
    if image.max() > 1:
        skeleton = skeleton * 255

    return skeleton.astype(np.uint8)


def detect_minutiae(skeleton):
    """
    Detect minutiae points in a thinned (skeletonized) fingerprint image.

    Returns:
        - endings: list of (x,y) coordinates of line endings (1 neighbor)
        - bifurcations: list of (x,y) coordinates of bifurcations (3 neighbors)
        - crossings: list of (x,y) coordinates of crossings (4+ neighbors)
    """
    # Make sure we're working with a binary image (0 and 1)
    if skeleton.max() > 1:
        skeleton = skeleton // 255

    # Lists to store minutiae
    endings = []
    bifurcations = []
    crossings = []

    # Get image dimensions
    rows, cols = skeleton.shape

    # Pad the image to handle border pixels
    padded = np.pad(skeleton, ((1, 1), (1, 1)), 'constant')

    # Scan through the image (excluding borders)
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if padded[i, j] == 1:  # If it's a ridge pixel
                # Get 8 neighbors
                p2 = int(padded[i - 1, j])
                p3 = int(padded[i - 1, j + 1])
                p4 = int(padded[i, j + 1])
                p5 = int(padded[i + 1, j + 1])
                p6 = int(padded[i + 1, j])
                p7 = int(padded[i + 1, j - 1])
                p8 = int(padded[i, j - 1])
                p9 = int(padded[i - 1, j - 1])

                # Count neighbors
                neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

                # Classify minutiae based on neighbor count
                if neighbor_count == 1:
                    endings.append((j - 1, i - 1))  # (x,y) format for drawing
                elif neighbor_count == 3:
                    bifurcations.append((j - 1, i - 1))
                elif neighbor_count >= 4:
                    crossings.append((j - 1, i - 1))

    return endings, bifurcations, crossings


def highlight_minutiae(image, endings, bifurcations, crossings):
    """
    Highlight different types of minutiae with different colors.

    - Endings: Red (0, 0, 255)
    - Bifurcations: Green (0, 255, 0)
    - Crossings: Blue (255, 0, 0)
    """
    # Convert grayscale image to color for highlighting
    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()

    # Define colors and sizes for different minutiae
    ending_color = (0, 0, 255)  # Red in BGR
    bifurcation_color = (0, 255, 0)  # Green in BGR
    crossing_color = (255, 0, 0)  # Blue in BGR

    # Draw circles around minutiae
    for point in endings:
        cv2.circle(result, point, 5, ending_color, -1)

    for point in bifurcations:
        cv2.circle(result, point, 5, bifurcation_color, -1)

    for point in crossings:
        cv2.circle(result, point, 5, crossing_color, -1)

    return result


def segment_fingerprint(image, block_size=16, threshold=25):
    """
    Segment fingerprint to isolate the central part and remove edge artifacts.
    More aggressive centralization to reduce false minutiae.

    Args:
        image: Grayscale fingerprint image
        block_size: Size of blocks for variance calculation
        threshold: Variance threshold for segmentation

    Returns:
        Segmented binary image (mask)
    """
    # Create empty mask
    rows, cols = image.shape
    mask = np.zeros((rows, cols), dtype=np.uint8)

    # Calculate mean and variance for each block
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # Get current block
            block = image[i:min(i + block_size, rows), j:min(j + block_size, cols)]

            # Calculate variance
            if block.size > 0:
                variance = np.var(block)

                # If variance is above threshold, include this block
                # Higher variance indicates ridge patterns (more texture)
                if variance > threshold:
                    mask[i:min(i + block_size, rows), j:min(j + block_size, cols)] = 255

    # Apply morphological operations to clean up the mask
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours and keep only the largest one (the fingerprint)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Create new mask with only the largest contour
        largest_mask = np.zeros_like(mask)
        cv2.drawContours(largest_mask, [contours[0]], 0, 255, -1)

        # Find the center of the fingerprint
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = cols // 2, rows // 2

        # Create a distance map from the center
        Y, X = np.ogrid[:rows, :cols]
        distance_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

        # Calculate radius that contains ~60% of the fingerprint area
        area = cv2.countNonZero(largest_mask)
        radius = np.sqrt(0.9 * area / np.pi)

        # Create a circular mask centered at the fingerprint center
        circle_mask = np.zeros_like(mask)
        cv2.circle(circle_mask, (center_x, center_y), int(radius), 255, -1)

        # Combine the largest contour mask with the circle mask (intersection)
        final_mask = cv2.bitwise_and(largest_mask, circle_mask)

        # Dilate slightly to ensure we don't cut off important features
        final_mask = cv2.dilate(final_mask, np.ones((15, 15), np.uint8), iterations=1)

        # Erode the edges more aggressively to remove boundary minutiae
        final_mask = cv2.erode(final_mask, np.ones((15, 15), np.uint8), iterations=1)

        return final_mask

    # If no contours found, return original mask
    return mask


def filter_edge_minutiae(mask, endings, bifurcations, crossings, distance_threshold=25):
    """
    Filter out minutiae that are too close to the edge of the segmentation mask.

    Args:
        mask: Binary segmentation mask
        endings: List of ending minutiae points (x,y)
        bifurcations: List of bifurcation minutiae points (x,y)
        crossings: List of crossing minutiae points (x,y)
        distance_threshold: Minimum distance from edge to keep a minutia point

    Returns:
        Filtered lists of endings, bifurcations, and crossings
    """
    # Create a distance transform of the inverse mask
    # This gives us the distance from each foreground pixel to the nearest background pixel
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    # Filter function for a list of minutiae
    def filter_minutiae(minutiae_list):
        filtered = []
        for x, y in minutiae_list:
            # Check if the point is within the mask
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
                # Get the distance to the nearest edge
                distance_to_edge = dist_transform[y, x]
                # Keep the minutia if it's far enough from the edge
                if distance_to_edge >= distance_threshold:
                    filtered.append((x, y))
        return filtered

    # Filter each type of minutiae
    filtered_endings = filter_minutiae(endings)
    filtered_bifurcations = filter_minutiae(bifurcations)
    filtered_crossings = filter_minutiae(crossings)

    return filtered_endings, filtered_bifurcations, filtered_crossings


# Update the process_fingerprint function to use the filter
def process_fingerprint(image, algorithm=ThinningAlgorithm.GUO_HALL):
    """
    Process a fingerprint image through thinning and minutiae detection.

    Args:
        image: Input image
        algorithm: Thinning algorithm to use (GUO_HALL or ZHANG_SUEN)

    Returns:
        original: Original binary image
        skeleton: Thinned image
        highlighted: Image with highlighted minutiae
        minutiae_data: Tuple of (endings, bifurcations, crossings) lists
        mask: Segmentation mask
    """
    # Ensure grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Image pre-processing
    # Enhance contrast
    gray = cv2.equalizeHist(gray)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Further enhance ridge-valley contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Segment the fingerprint (extract core area)
    mask = segment_fingerprint(gray, block_size=16, threshold=25)

    # Binarize with adaptive thresholding for better results
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Apply mask to binary image
    binary = cv2.bitwise_and(binary, binary, mask=mask)

    # Remove small noise with morphological operations
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Apply thinning based on selected algorithm
    if algorithm == ThinningAlgorithm.ZHANG_SUEN:
        skeleton = zhang_suen_thinning(binary)
    else:  # Default to Guo-Hall
        skeleton = guo_hall_thinning(binary)

    # Apply mask to skeleton to remove edge artifacts
    skeleton = cv2.bitwise_and(skeleton, skeleton, mask=mask)

    # Perform pruning to remove small branches (false minutiae)
    pruned_skeleton = prune_skeleton(skeleton, length_threshold=5)

    # Detect minutiae
    endings, bifurcations, crossings = detect_minutiae(pruned_skeleton)

    # Filter minutiae to remove those too close to the edges
    endings, bifurcations, crossings = filter_edge_minutiae(mask, endings, bifurcations, crossings,
                                                            distance_threshold=25)

    # Highlight minutiae
    highlighted = highlight_minutiae(pruned_skeleton, endings, bifurcations, crossings)

    # Return the original mask for visualization too
    return binary, pruned_skeleton, highlighted, (endings, bifurcations, crossings), mask


def prune_skeleton(skeleton, length_threshold=5):
    """
    Remove short branches from skeleton that are likely false minutiae.

    Args:
        skeleton: Skeletonized binary image
        length_threshold: Maximum length of branches to remove

    Returns:
        Pruned skeleton image
    """
    # Make a copy of the skeleton
    pruned = skeleton.copy()

    # Find endpoints
    rows, cols = pruned.shape
    endpoint_map = np.zeros_like(pruned)

    # Pad the image to handle border pixels
    padded = np.pad(pruned, ((1, 1), (1, 1)), 'constant')

    # Detect endpoints (pixels with only one neighbor)
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if padded[i, j] == 1:  # If it's a ridge pixel
                # Get 8 neighbors
                p2 = int(padded[i - 1, j])
                p3 = int(padded[i - 1, j + 1])
                p4 = int(padded[i, j + 1])
                p5 = int(padded[i + 1, j + 1])
                p6 = int(padded[i + 1, j])
                p7 = int(padded[i + 1, j - 1])
                p8 = int(padded[i, j - 1])
                p9 = int(padded[i - 1, j - 1])

                # Count neighbors
                neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

                # If it has exactly one neighbor, it's an endpoint
                if neighbor_count == 1:
                    endpoint_map[i - 1, j - 1] = 1

    # Now for each endpoint, trace the branch and remove it if it's shorter than threshold
    for i in range(rows):
        for j in range(cols):
            if endpoint_map[i, j] == 1:
                # Start tracing from this endpoint
                branch_len = 0
                current_i, current_j = i, j
                branch_points = [(current_i, current_j)]

                # Continue tracing until we reach a branch point or another endpoint
                while True:
                    # Look at the 8-neighborhood of the current pixel
                    neighbors = []
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue  # Skip the center pixel

                            ni, nj = current_i + di, current_j + dj
                            if 0 <= ni < rows and 0 <= nj < cols and pruned[ni, nj] == 1:
                                neighbors.append((ni, nj))

                    # If we have exactly one neighbor that's not already in our branch,
                    # continue tracing in that direction
                    new_neighbors = [n for n in neighbors if n not in branch_points]

                    if len(new_neighbors) == 1:
                        current_i, current_j = new_neighbors[0]
                        branch_points.append((current_i, current_j))
                        branch_len += 1
                    else:
                        # We've reached either a branch point or another endpoint
                        break

                # If branch is shorter than threshold, remove it
                if branch_len < length_threshold:
                    for point in branch_points:
                        pruned[point] = 0

    return pruned


# Main GUI application
def HELP_window(message):
    message_window = tk.Toplevel()
    message_window.title("Info")
    message_window.geometry("500x350")
    label = ttk.Label(message_window, text=message, wraplength=400)
    label.pack(pady=20)
    close_button = ttk.Button(message_window, text="Zamknij", command=message_window.destroy)
    close_button.pack(pady=10)


def kill_UI():
    for widget in window.winfo_children():
        widget.destroy()


def display_results(binary, skeleton, highlighted, minutiae, mask):
    """Display the results in the UI"""
    kill_UI()

    # Unpack minutiae
    endings, bifurcations, crossings = minutiae

    # Create a frame for displaying images
    display_frame = ttk.Frame(window)
    display_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Convert OpenCV images to PIL format for Tkinter
    binary_pil = Image.fromarray(binary)
    skeleton_pil = Image.fromarray(skeleton)
    highlighted_pil = Image.fromarray(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask)

    # Resize images for display - bigger now!
    max_height = 300
    binary_pil.thumbnail((400, max_height))
    skeleton_pil.thumbnail((400, max_height))
    highlighted_pil.thumbnail((400, max_height))
    mask_pil.thumbnail((400, max_height))

    # Convert to PhotoImage for tkinter
    binary_tk = ImageTk.PhotoImage(binary_pil)
    skeleton_tk = ImageTk.PhotoImage(skeleton_pil)
    highlighted_tk = ImageTk.PhotoImage(highlighted_pil)
    mask_tk = ImageTk.PhotoImage(mask_pil)

    # Store references to prevent garbage collection
    display_frame.binary_tk = binary_tk
    display_frame.skeleton_tk = skeleton_tk
    display_frame.highlighted_tk = highlighted_tk
    display_frame.mask_tk = mask_tk

    # Create labels for each image - reorganize to fit 4 images in 2 rows
    binary_label = ttk.Label(display_frame, image=binary_tk)
    binary_label.grid(row=0, column=0, padx=5, pady=5)
    ttk.Label(display_frame, text="Binary Image").grid(row=1, column=0)

    mask_label = ttk.Label(display_frame, image=mask_tk)
    mask_label.grid(row=0, column=1, padx=5, pady=0)
    ttk.Label(display_frame, text="Segmentation Mask").grid(row=1, column=1)

    skeleton_label = ttk.Label(display_frame, image=skeleton_tk)
    skeleton_label.grid(row=2, column=0, padx=5, pady=0)
    ttk.Label(display_frame, text="Skeleton").grid(row=3, column=0)

    highlighted_label = ttk.Label(display_frame, image=highlighted_tk)
    highlighted_label.grid(row=2, column=1, padx=5, pady=0)
    ttk.Label(display_frame, text="Minutiae").grid(row=3, column=1)

    # Display minutiae statistics
    stats_frame = ttk.LabelFrame(window, text="Minutiae Statistics")
    stats_frame.pack(fill="x", padx=10, pady=5)

    minutiae_text = f"Endings: {len(endings)}\nBifurcations: {len(bifurcations)}\nCrossings: {len(crossings)}\n"
    minutiae_text += f"Total: {len(endings) + len(bifurcations) + len(crossings)}"

    stats_label = ttk.Label(stats_frame, text=minutiae_text)
    stats_label.pack(padx=10, pady=5)

    # Save button
    save_frame = ttk.Frame(window)
    save_frame.pack(fill="x", padx=10, pady=5)

    save_button = ttk.Button(
        save_frame,
        text='Save Results',
        command=lambda: save_results(binary, skeleton, highlighted, mask)
    )
    save_button.pack(side="left", padx=5)

    # New analysis button
    new_button = ttk.Button(
        save_frame,
        text='New Analysis',
        command=reset_UI
    )
    new_button.pack(side="right", padx=5)


def save_results(binary, skeleton, highlighted, mask=None):
    """Save the processed images to the timestamped folder"""
    try:
        # Create the timestamp folder if it doesn't exist
        if not os.path.exists(timestamped_folder_path):
            os.makedirs(timestamped_folder_path)

        # Save images
        cv2.imwrite(os.path.join(timestamped_folder_path, "binary.png"), binary)
        cv2.imwrite(os.path.join(timestamped_folder_path, "skeleton.png"), skeleton)
        cv2.imwrite(os.path.join(timestamped_folder_path, "minutiae.png"), highlighted)
        if mask is not None:
            cv2.imwrite(os.path.join(timestamped_folder_path, "mask.png"), mask)

        # Save minutiae data as a figure
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(binary, cmap='gray')
        plt.title("Binary Image")

        if mask is not None:
            plt.subplot(2, 2, 2)
            plt.imshow(mask, cmap='gray')
            plt.title("Segmentation Mask")

        plt.subplot(2, 2, 3)
        plt.imshow(skeleton, cmap='gray')
        plt.title("Skeleton")

        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
        plt.title("Detected Minutiae")

        plt.tight_layout()
        plt.savefig(os.path.join(timestamped_folder_path, "results.png"))

        showinfo(title="Success", message=f"Results saved to {timestamped_folder_path}")
    except Exception as e:
        showinfo(title="Error", message=f"Failed to save results: {e}")


def reset_UI():
    """Reset UI to initial state"""
    kill_UI()
    setup_initial_UI()


def setup_initial_UI():
    """Set up the initial UI with the file open button"""
    open_button1 = ttk.Button(
        window,
        text='Otworz plik',
        command=select_file
    )
    open_button1.place(x=250, y=400)


def select_file():
    """Select and process a fingerprint image"""
    kill_UI()
    global full_path, dataframe, filename, image_location, image_name, timestamped_folder_path

    filetypes = (
        ('jpg files', '*.jpg'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Wybierz plik',
        initialdir='/',
        filetypes=filetypes
    )

    if filename:
        try:
            # Process path information
            image_location, image_name = os.path.split(filename)
            full_path = os.path.join(image_location, image_name).replace('\\', '/')

            # Create timestamp folder
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            timestamped_folder_path = os.path.join(image_location, timestamp)
            timestamped_folder_path = os.path.normpath(timestamped_folder_path).replace('\\', '/')

            # Load and process the image
            cv_image = cv2.imread(filename)
            if cv_image is None:
                showinfo(title="Error", message="Could not load image with OpenCV. Trying with PIL...")
                # Try with PIL and convert to OpenCV format
                pil_image = Image.open(filename)
                cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # Process the fingerprint
            binary, skeleton, highlighted, minutiae, mask = process_fingerprint(cv_image, ThinningAlgorithm.GUO_HALL)

            # Display results
            display_results(binary, skeleton, highlighted, minutiae, mask)

        except Exception as e:
            showinfo(title="Blad", message=f"{e}")
            reset_UI()


def setup_initial_UI():
    """Set up the initial UI with file open and comparison buttons"""
    frame = ttk.Frame(window)
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    title_label = ttk.Label(frame, text="Fingerprint Analysis System", font=("Arial", 16))
    title_label.pack(pady=20)

    # Button to open a single file
    open_button = ttk.Button(
        frame,
        text='Open Fingerprint Image',
        command=select_file
    )
    open_button.pack(pady=10)

    # Button to compare two files
    compare_button = ttk.Button(
        frame,
        text='Compare Two Fingerprints',
        command=lambda: select_file(compare=True)
    )
    compare_button.pack(pady=10)

    # Button to compare algorithms
    compare_algo_button = ttk.Button(
        frame,
        text='Compare Algorithms (Same Image)',
        command=compare_algorithms
    )
    compare_algo_button.pack(pady=10)

    # Help button
    help_button = ttk.Button(
        frame,
        text='Help',
        command=lambda: HELP_window("This application analyzes fingerprint images to detect minutiae.\n\n"
                                    "- Open Image: Process a single fingerprint image\n"
                                    "- Compare Two: Compare different fingerprint images\n"
                                    "- Compare Algorithms: Compare Zhang-Suen vs Guo-Hall on same image\n\n"
                                    "Minutiae Types:\n"
                                    "- Red: Endings\n"
                                    "- Green: Bifurcations\n"
                                    "- Blue: Crossings")
    )
    help_button.pack(pady=20)


def select_file(compare=False):
    """Select and process a fingerprint image"""
    global comparison_mode
    comparison_mode = compare

    kill_UI()

    filetypes = (
        ('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff'),
        ('All files', '*.*')
    )

    if compare:
        # For comparison, we need two files
        showinfo(title="Compare Mode", message="Select the FIRST fingerprint image")
        filename1 = fd.askopenfilename(
            title='Select first image',
            initialdir='/',
            filetypes=filetypes
        )

        if not filename1:
            reset_UI()
            return

        showinfo(title="Compare Mode", message="Select the SECOND fingerprint image")
        filename2 = fd.askopenfilename(
            title='Select second image',
            initialdir='/',
            filetypes=filetypes
        )

        if not filename2:
            reset_UI()
            return

        process_comparison(filename1, filename2)
    else:
        # Single file mode
        filename = fd.askopenfilename(
            title='Select fingerprint image',
            initialdir='/',
            filetypes=filetypes
        )

        if not filename:
            reset_UI()
            return

        process_single_file(filename)


def process_single_file(filename):
    """Process a single fingerprint image with algorithm selection"""
    global full_path, dataframe, image_location, image_name, timestamped_folder_path

    try:
        # Process path information
        image_location, image_name = os.path.split(filename)
        full_path = os.path.join(image_location, image_name).replace('\\', '/')

        # Create timestamp folder
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        timestamped_folder_path = os.path.join(image_location, timestamp)
        timestamped_folder_path = os.path.normpath(timestamped_folder_path).replace('\\', '/')

        # Load the image
        cv_image = cv2.imread(filename)
        if cv_image is None:
            showinfo(title="Error", message="Could not load image with OpenCV. Trying with PIL...")
            # Try with PIL and convert to OpenCV format
            pil_image = Image.open(filename)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Create algorithm selection frame
        algo_frame = ttk.Frame(window)
        algo_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(algo_frame, text="Select Thinning Algorithm:").pack(side="left", padx=5)

        # Variable to store the selected algorithm
        algo_var = tk.StringVar(value="guo-hall")

        # Radio buttons for algorithm selection
        ttk.Radiobutton(
            algo_frame,
            text="Guo-Hall",
            variable=algo_var,
            value="guo-hall"
        ).pack(side="left", padx=10)

        ttk.Radiobutton(
            algo_frame,
            text="Zhang-Suen",
            variable=algo_var,
            value="zhang-suen"
        ).pack(side="left", padx=10)

        # Process button
        process_button = ttk.Button(
            algo_frame,
            text="Process Image",
            command=lambda: process_with_algorithm(cv_image, algo_var.get())
        )
        process_button.pack(side="right", padx=10)

        # Back button
        back_button = ttk.Button(
            algo_frame,
            text="Back",
            command=reset_UI
        )
        back_button.pack(side="right", padx=10)

        # Display original image preview
        preview_frame = ttk.Frame(window)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Resize for preview
        if len(cv_image.shape) == 3:
            preview_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            preview_img = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

        pil_img = Image.fromarray(preview_img)
        pil_img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(pil_img)

        # Store reference to prevent garbage collection
        preview_frame.tk_img = tk_img

        img_label = ttk.Label(preview_frame, image=tk_img)
        img_label.pack(pady=10)
        ttk.Label(preview_frame, text=f"Original Image: {image_name}").pack()

    except Exception as e:
        showinfo(title="Error", message=f"{e}")
        reset_UI()


def process_with_algorithm(cv_image, algo_name):
    """Process the image with the selected algorithm"""
    global current_algo, results_cache

    try:
        # Set the algorithm
        if algo_name == "zhang-suen":
            algorithm = ThinningAlgorithm.ZHANG_SUEN
        else:
            algorithm = ThinningAlgorithm.GUO_HALL

        current_algo = algorithm

        # Process the fingerprint
        binary, skeleton, highlighted, minutiae, mask = process_fingerprint(cv_image, algorithm)

        # Cache the results
        results_cache[algo_name] = (binary, skeleton, highlighted, minutiae, mask)

        # Display results
        display_results(binary, skeleton, highlighted, minutiae, mask)

    except Exception as e:
        showinfo(title="Error", message=f"{e}")


def compare_algorithms():
    """Compare two algorithms on the same image"""
    kill_UI()

    filetypes = (
        ('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Select fingerprint image',
        initialdir='/',
        filetypes=filetypes
    )

    if not filename:
        reset_UI()
        return

    try:
        # Process path information
        global full_path, image_location, image_name, timestamped_folder_path
        image_location, image_name = os.path.split(filename)
        full_path = os.path.join(image_location, image_name).replace('\\', '/')

        # Create timestamp folder
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        timestamped_folder_path = os.path.join(image_location, timestamp)
        timestamped_folder_path = os.path.normpath(timestamped_folder_path).replace('\\', '/')

        # Load the image
        cv_image = cv2.imread(filename)
        if cv_image is None:
            showinfo(title="Error", message="Could not load image with OpenCV. Trying with PIL...")
            # Try with PIL and convert to OpenCV format
            pil_image = Image.open(filename)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Process with both algorithms
        binary1, skeleton1, highlighted1, minutiae1, mask1 = process_fingerprint(cv_image, ThinningAlgorithm.GUO_HALL)
        binary2, skeleton2, highlighted2, minutiae2, mask2 = process_fingerprint(cv_image, ThinningAlgorithm.ZHANG_SUEN)

        # Display comparison
        display_comparison(
            [binary1, skeleton1, highlighted1, minutiae1, mask1],
            [binary2, skeleton2, highlighted2, minutiae2, mask2],
            "Guo-Hall Algorithm",
            "Zhang-Suen Algorithm"
        )

    except Exception as e:
        showinfo(title="Error", message=f"{e}")
        reset_UI()


def process_comparison(filename1, filename2):
    """Process and compare two different fingerprint images"""
    try:
        # Process path information for the first image
        global full_path, image_location, image_name, timestamped_folder_path
        image_location, image_name = os.path.split(filename1)
        full_path = os.path.join(image_location, image_name).replace('\\', '/')

        # Create timestamp folder
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        timestamped_folder_path = os.path.join(image_location, timestamp)
        timestamped_folder_path = os.path.normpath(timestamped_folder_path).replace('\\', '/')

        # Create algorithm selection frame
        algo_frame = ttk.Frame(window)
        algo_frame.pack(fill="x", padx=10, pady=10)

        # Create frame for first image algorithm selection
        img1_frame = ttk.LabelFrame(algo_frame, text=f"Algorithm for {os.path.basename(filename1)}")
        img1_frame.pack(side="left", padx=10, pady=5, fill="x", expand=True)

        # Variable to store the selected algorithm for first image
        algo_var1 = tk.StringVar(value="guo-hall")

        # Radio buttons for algorithm selection for first image
        ttk.Radiobutton(
            img1_frame,
            text="Guo-Hall",
            variable=algo_var1,
            value="guo-hall"
        ).pack(side="left", padx=10)

        ttk.Radiobutton(
            img1_frame,
            text="Zhang-Suen",
            variable=algo_var1,
            value="zhang-suen"
        ).pack(side="left", padx=10)

        # Create frame for second image algorithm selection
        img2_frame = ttk.LabelFrame(algo_frame, text=f"Algorithm for {os.path.basename(filename2)}")
        img2_frame.pack(side="left", padx=10, pady=5, fill="x", expand=True)

        # Variable to store the selected algorithm for second image
        algo_var2 = tk.StringVar(value="guo-hall")

        # Radio buttons for algorithm selection for second image
        ttk.Radiobutton(
            img2_frame,
            text="Guo-Hall",
            variable=algo_var2,
            value="guo-hall"
        ).pack(side="left", padx=10)

        ttk.Radiobutton(
            img2_frame,
            text="Zhang-Suen",
            variable=algo_var2,
            value="zhang-suen"
        ).pack(side="left", padx=10)

        # Button frame
        button_frame = ttk.Frame(algo_frame)
        button_frame.pack(side="right", padx=10)

        # Process button
        process_button = ttk.Button(
            button_frame,
            text="Process Images",
            command=lambda: compare_two_images(filename1, filename2, algo_var1.get(), algo_var2.get())
        )
        process_button.pack(side="top", pady=5)

        # Back button
        back_button = ttk.Button(
            button_frame,
            text="Back",
            command=reset_UI
        )
        back_button.pack(side="bottom", pady=5)

        # Display original image previews
        preview_frame = ttk.Frame(window)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Load first image
        cv_image1 = cv2.imread(filename1)
        if cv_image1 is None:
            pil_image = Image.open(filename1)
            cv_image1 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Load second image
        cv_image2 = cv2.imread(filename2)
        if cv_image2 is None:
            pil_image = Image.open(filename2)
            cv_image2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Create a frame for images side by side
        images_frame = ttk.Frame(preview_frame)
        images_frame.pack(fill="both", expand=True)

        # Process first image for preview
        if len(cv_image1.shape) == 3:
            preview_img1 = cv2.cvtColor(cv_image1, cv2.COLOR_BGR2RGB)
        else:
            preview_img1 = cv2.cvtColor(cv_image1, cv2.COLOR_GRAY2RGB)

        # Process second image for preview
        if len(cv_image2.shape) == 3:
            preview_img2 = cv2.cvtColor(cv_image2, cv2.COLOR_BGR2RGB)
        else:
            preview_img2 = cv2.cvtColor(cv_image2, cv2.COLOR_GRAY2RGB)

        # Resize and convert to Tkinter format
        pil_img1 = Image.fromarray(preview_img1)
        pil_img1.thumbnail((300, 300))
        tk_img1 = ImageTk.PhotoImage(pil_img1)

        pil_img2 = Image.fromarray(preview_img2)
        pil_img2.thumbnail((300, 300))
        tk_img2 = ImageTk.PhotoImage(pil_img2)

        # Store references to prevent garbage collection
        images_frame.tk_img1 = tk_img1
        images_frame.tk_img2 = tk_img2

        # Display images side by side
        left_frame = ttk.Frame(images_frame)
        left_frame.pack(side="left", padx=10)

        right_frame = ttk.Frame(images_frame)
        right_frame.pack(side="right", padx=10)

        # First image
        img_label1 = ttk.Label(left_frame, image=tk_img1)
        img_label1.pack(pady=10)
        ttk.Label(left_frame, text=f"Image 1: {os.path.basename(filename1)}").pack()

        # Second image
        img_label2 = ttk.Label(right_frame, image=tk_img2)
        img_label2.pack(pady=10)
        ttk.Label(right_frame, text=f"Image 2: {os.path.basename(filename2)}").pack()

    except Exception as e:
        showinfo(title="Error", message=f"{e}")
        reset_UI()


def compare_two_images(filename1, filename2, algo_name1, algo_name2):
    """Process and compare two different images with potentially different algorithms"""
    try:
        # Set the algorithms
        if algo_name1 == "zhang-suen":
            algorithm1 = ThinningAlgorithm.ZHANG_SUEN
        else:
            algorithm1 = ThinningAlgorithm.GUO_HALL

        if algo_name2 == "zhang-suen":
            algorithm2 = ThinningAlgorithm.ZHANG_SUEN
        else:
            algorithm2 = ThinningAlgorithm.GUO_HALL

        # Load first image
        cv_image1 = cv2.imread(filename1)
        if cv_image1 is None:
            pil_image = Image.open(filename1)
            cv_image1 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Load second image
        cv_image2 = cv2.imread(filename2)
        if cv_image2 is None:
            pil_image = Image.open(filename2)
            cv_image2 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Process both images with their respective algorithms
        binary1, skeleton1, highlighted1, minutiae1, mask1 = process_fingerprint(cv_image1, algorithm1)
        binary2, skeleton2, highlighted2, minutiae2, mask2 = process_fingerprint(cv_image2, algorithm2)

        # Get algorithm names for display
        algo_display1 = "Zhang-Suen" if algorithm1 == ThinningAlgorithm.ZHANG_SUEN else "Guo-Hall"
        algo_display2 = "Zhang-Suen" if algorithm2 == ThinningAlgorithm.ZHANG_SUEN else "Guo-Hall"

        # Create button frame for navigation controls
        nav_button_frame = ttk.Frame(window)
        nav_button_frame.pack(side="bottom", fill="x", padx=10, pady=10)

        # New analysis button in the navigation frame
        new_button = ttk.Button(
            nav_button_frame,
            text='New Analysis',
            command=reset_UI
        )
        new_button.pack(side="right", padx=5)

        # Display comparison with algorithm info
        display_comparison(
            [binary1, skeleton1, highlighted1, minutiae1, mask1],
            [binary2, skeleton2, highlighted2, minutiae2, mask2],
            f"{os.path.basename(filename1)} ({algo_display1})",
            f"{os.path.basename(filename2)} ({algo_display2})"
        )

    except Exception as e:
        showinfo(title="Error", message=f"{e}")
        reset_UI()

def display_comparison(results1, results2, title1, title2):
    """Display comparison of two sets of results side by side"""
    kill_UI()

    # Unpack results
    binary1, skeleton1, highlighted1, minutiae1, mask1 = results1
    binary2, skeleton2, highlighted2, minutiae2, mask2 = results2

    # Unpack minutiae
    endings1, bifurcations1, crossings1 = minutiae1
    endings2, bifurcations2, crossings2 = minutiae2

    # Create a frame for displaying images
    display_frame = ttk.Frame(window)
    display_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Title for comparison
    ttk.Label(display_frame, text="Fingerprint Comparison", font=("Arial", 14)).grid(row=0, column=0, columnspan=6, pady=10)

    # Create top and bottom frames
    top_frame = ttk.LabelFrame(display_frame, text=title1)
    top_frame.grid(row=1, column=0, columnspan=6, padx=10, pady=5, sticky="nsew")

    bottom_frame = ttk.LabelFrame(display_frame, text=title2)
    bottom_frame.grid(row=2, column=0, columnspan=6, padx=10, pady=5, sticky="nsew")

    # Convert OpenCV images to PIL format for Tkinter
    binary_pil1 = Image.fromarray(binary1)
    skeleton_pil1 = Image.fromarray(skeleton1)
    highlighted_pil1 = Image.fromarray(cv2.cvtColor(highlighted1, cv2.COLOR_BGR2RGB))

    binary_pil2 = Image.fromarray(binary2)
    skeleton_pil2 = Image.fromarray(skeleton2)
    highlighted_pil2 = Image.fromarray(cv2.cvtColor(highlighted2, cv2.COLOR_BGR2RGB))

    # Resize images for display
    max_width = 300
    binary_pil1.thumbnail((max_width, 250))
    skeleton_pil1.thumbnail((max_width, 250))
    highlighted_pil1.thumbnail((max_width, 250))

    binary_pil2.thumbnail((max_width, 250))
    skeleton_pil2.thumbnail((max_width, 250))
    highlighted_pil2.thumbnail((max_width, 250))

    # Convert to PhotoImage for tkinter
    binary_tk1 = ImageTk.PhotoImage(binary_pil1)
    skeleton_tk1 = ImageTk.PhotoImage(skeleton_pil1)
    highlighted_tk1 = ImageTk.PhotoImage(highlighted_pil1)

    binary_tk2 = ImageTk.PhotoImage(binary_pil2)
    skeleton_tk2 = ImageTk.PhotoImage(skeleton_pil2)
    highlighted_tk2 = ImageTk.PhotoImage(highlighted_pil2)

    # Store references to prevent garbage collection
    display_frame.binary_tk1 = binary_tk1
    display_frame.skeleton_tk1 = skeleton_tk1
    display_frame.highlighted_tk1 = highlighted_tk1
    display_frame.binary_tk2 = binary_tk2
    display_frame.skeleton_tk2 = skeleton_tk2
    display_frame.highlighted_tk2 = highlighted_tk2

    # Top row images (first fingerprint)
    binary_label1 = ttk.Label(top_frame, image=binary_tk1)
    binary_label1.grid(row=0, column=0, padx=5, pady=5)
    ttk.Label(top_frame, text="Binary Image").grid(row=1, column=0)

    skeleton_label1 = ttk.Label(top_frame, image=skeleton_tk1)
    skeleton_label1.grid(row=0, column=1, padx=5, pady=5)
    ttk.Label(top_frame, text="Skeleton").grid(row=1, column=1)

    highlighted_label1 = ttk.Label(top_frame, image=highlighted_tk1)
    highlighted_label1.grid(row=0, column=2, padx=5, pady=5)
    ttk.Label(top_frame, text="Minutiae").grid(row=1, column=2)

    # Bottom row images (second fingerprint)
    binary_label2 = ttk.Label(bottom_frame, image=binary_tk2)
    binary_label2.grid(row=0, column=0, padx=5, pady=5)
    ttk.Label(bottom_frame, text="Binary Image").grid(row=1, column=0)

    skeleton_label2 = ttk.Label(bottom_frame, image=skeleton_tk2)
    skeleton_label2.grid(row=0, column=1, padx=5, pady=5)
    ttk.Label(bottom_frame, text="Skeleton").grid(row=1, column=1)

    highlighted_label2 = ttk.Label(bottom_frame, image=highlighted_tk2)
    highlighted_label2.grid(row=0, column=2, padx=5, pady=5)
    ttk.Label(bottom_frame, text="Minutiae").grid(row=1, column=2)

    # Display minutiae statistics
    stats_frame = ttk.LabelFrame(window, text="Minutiae Statistics")
    stats_frame.pack(fill="x", padx=10, pady=5)

    # Create frame for statistics
    stats_grid = ttk.Frame(stats_frame)
    stats_grid.pack(padx=10, pady=10)

    # Column headers
    ttk.Label(stats_grid, text="", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=5, pady=2)
    ttk.Label(stats_grid, text=title1, font=("Arial", 10, "bold")).grid(row=0, column=1, padx=5, pady=2)
    ttk.Label(stats_grid, text=title2, font=("Arial", 10, "bold")).grid(row=0, column=2, padx=5, pady=2)
    ttk.Label(stats_grid, text="Difference", font=("Arial", 10, "bold")).grid(row=0, column=3, padx=5, pady=2)

    # Endings
    ttk.Label(stats_grid, text="Endings:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
    ttk.Label(stats_grid, text=f"{len(endings1)}").grid(row=1, column=1, padx=5, pady=2)
    ttk.Label(stats_grid, text=f"{len(endings2)}").grid(row=1, column=2, padx=5, pady=2)
    ttk.Label(stats_grid, text=f"{len(endings1) - len(endings2)}").grid(row=1, column=3, padx=5, pady=2)

    # Bifurcations
    ttk.Label(stats_grid, text="Bifurcations:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
    ttk.Label(stats_grid, text=f"{len(bifurcations1)}").grid(row=2, column=1, padx=5, pady=2)
    ttk.Label(stats_grid, text=f"{len(bifurcations2)}").grid(row=2, column=2, padx=5, pady=2)
    ttk.Label(stats_grid, text=f"{len(bifurcations1) - len(bifurcations2)}").grid(row=2, column=3, padx=5, pady=2)

    # Crossings
    ttk.Label(stats_grid, text="Crossings:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
    ttk.Label(stats_grid, text=f"{len(crossings1)}").grid(row=3, column=1, padx=5, pady=2)
    ttk.Label(stats_grid, text=f"{len(crossings2)}").grid(row=3, column=2, padx=5, pady=2)
    ttk.Label(stats_grid, text=f"{len(crossings1) - len(crossings2)}").grid(row=3, column=3, padx=5, pady=2)

    # Total
    ttk.Label(stats_grid, text="Total:", font=("Arial", 10, "bold")).grid(row=4, column=0, padx=5, pady=2, sticky="w")
    total1 = len(endings1) + len(bifurcations1) + len(crossings1)
    total2 = len(endings2) + len(bifurcations2) + len(crossings2)
    ttk.Label(stats_grid, text=f"{total1}", font=("Arial", 10, "bold")).grid(row=4, column=1, padx=5, pady=2)
    ttk.Label(stats_grid, text=f"{total2}", font=("Arial", 10, "bold")).grid(row=4, column=2, padx=5, pady=2)
    ttk.Label(stats_grid, text=f"{total1 - total2}", font=("Arial", 10, "bold")).grid(row=4, column=3, padx=5, pady=2)

    # Button frame
    button_frame = ttk.Frame(window)
    button_frame.pack(fill="x", padx=10, pady=5)

    # Save button
    save_button = ttk.Button(
        button_frame,
        text='Save Comparison',
        command=lambda: save_comparison(binary1, skeleton1, highlighted1, binary2, skeleton2, highlighted2, title1,
                                        title2)
    )
    save_button.pack(side="left", padx=5)

    # New analysis button
    new_button = ttk.Button(
        button_frame,
        text='New Analysis',
        command=reset_UI
    )
    new_button.pack(side="right", padx=5)

def save_comparison(binary1, skeleton1, highlighted1, binary2, skeleton2, highlighted2, title1, title2):
    """Save the comparison results to the timestamped folder"""
    try:
        # Create the timestamp folder if it doesn't exist
        if not os.path.exists(timestamped_folder_path):
            os.makedirs(timestamped_folder_path)

        # Save individual images
        cv2.imwrite(os.path.join(timestamped_folder_path, f"{title1}_binary.png"), binary1)
        cv2.imwrite(os.path.join(timestamped_folder_path, f"{title1}_skeleton.png"), skeleton1)
        cv2.imwrite(os.path.join(timestamped_folder_path, f"{title1}_minutiae.png"), highlighted1)

        cv2.imwrite(os.path.join(timestamped_folder_path, f"{title2}_binary.png"), binary2)
        cv2.imwrite(os.path.join(timestamped_folder_path, f"{title2}_skeleton.png"), skeleton2)
        cv2.imwrite(os.path.join(timestamped_folder_path, f"{title2}_minutiae.png"), highlighted2)

        # Save comparison as a figure
        plt.figure(figsize=(12, 12))

        # First row - Binary images
        plt.subplot(3, 2, 1)
        plt.imshow(binary1, cmap='gray')
        plt.title(f"{title1} - Binary")

        plt.subplot(3, 2, 2)
        plt.imshow(binary2, cmap='gray')
        plt.title(f"{title2} - Binary")

        # Second row - Skeletons
        plt.subplot(3, 2, 3)
        plt.imshow(skeleton1, cmap='gray')
        plt.title(f"{title1} - Skeleton")

        plt.subplot(3, 2, 4)
        plt.imshow(skeleton2, cmap='gray')
        plt.title(f"{title2} - Skeleton")

        # Third row - Minutiae
        plt.subplot(3, 2, 5)
        plt.imshow(cv2.cvtColor(highlighted1, cv2.COLOR_BGR2RGB))
        plt.title(f"{title1} - Minutiae")

        plt.subplot(3, 2, 6)
        plt.imshow(cv2.cvtColor(highlighted2, cv2.COLOR_BGR2RGB))
        plt.title(f"{title2} - Minutiae")

        plt.tight_layout()
        plt.savefig(os.path.join(timestamped_folder_path, "comparison.png"))

        showinfo(title="Success", message=f"Comparison saved to {timestamped_folder_path}")
    except Exception as e:
        showinfo(title="Error", message=f"Failed to save comparison: {e}")


def display_results(binary, skeleton, highlighted, minutiae, mask):
    """Display the results in the UI"""
    kill_UI()

    # Unpack minutiae
    endings, bifurcations, crossings = minutiae

    # Create a frame for displaying images
    display_frame = ttk.Frame(window)
    display_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # Title
    ttk.Label(display_frame, text="Fingerprint Analysis Results", font=("Arial", 14)).grid(row=0, column=0,
                                                                                           columnspan=3, pady=10)

    # Convert OpenCV images to PIL format for Tkinter
    binary_pil = Image.fromarray(binary)
    skeleton_pil = Image.fromarray(skeleton)
    highlighted_pil = Image.fromarray(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask) if mask is not None else None

    # Resize images for display
    max_height = 250
    binary_pil.thumbnail((300, max_height))
    skeleton_pil.thumbnail((300, max_height))
    highlighted_pil.thumbnail((300, max_height))
    if mask_pil:
        mask_pil.thumbnail((300, max_height))

    # Convert to PhotoImage for tkinter
    binary_tk = ImageTk.PhotoImage(binary_pil)
    skeleton_tk = ImageTk.PhotoImage(skeleton_pil)
    highlighted_tk = ImageTk.PhotoImage(highlighted_pil)
    mask_tk = ImageTk.PhotoImage(mask_pil) if mask_pil else None

    # Store references to prevent garbage collection
    display_frame.binary_tk = binary_tk
    display_frame.skeleton_tk = skeleton_tk
    display_frame.highlighted_tk = highlighted_tk
    if mask_tk:
        display_frame.mask_tk = mask_tk

    # Display images
    binary_label = ttk.Label(display_frame, image=binary_tk)
    binary_label.grid(row=1, column=0, padx=10, pady=5)
    ttk.Label(display_frame, text="Binary Image").grid(row=2, column=0)

    skeleton_label = ttk.Label(display_frame, image=skeleton_tk)
    skeleton_label.grid(row=1, column=1, padx=10, pady=5)
    ttk.Label(display_frame, text="Skeleton").grid(row=2, column=1)

    highlighted_label = ttk.Label(display_frame, image=highlighted_tk)
    highlighted_label.grid(row=1, column=2, padx=10, pady=5)
    ttk.Label(display_frame, text="Minutiae").grid(row=2, column=2)

    if mask_tk:
        mask_label = ttk.Label(display_frame, image=mask_tk)
        mask_label.grid(row=3, column=0, padx=10, pady=5)
        ttk.Label(display_frame, text="ROI Mask").grid(row=4, column=0)

    # Display minutiae statistics
    stats_frame = ttk.LabelFrame(window, text="Minutiae Statistics")
    stats_frame.pack(fill="x", padx=10, pady=0)

    # Create frame for statistics
    stats_grid = ttk.Frame(stats_frame)
    stats_grid.pack(padx=10, pady=0)

    # Add statistics
    ttk.Label(stats_grid, text="Endings:", font=("Arial", 10)).grid(row=0, column=0, padx=5, pady=2, sticky="w")
    ttk.Label(stats_grid, text=f"{len(endings)}").grid(row=0, column=1, padx=5, pady=2)

    ttk.Label(stats_grid, text="Bifurcations:", font=("Arial", 10)).grid(row=1, column=0, padx=5, pady=2, sticky="w")
    ttk.Label(stats_grid, text=f"{len(bifurcations)}").grid(row=1, column=1, padx=5, pady=2)

    ttk.Label(stats_grid, text="Crossings:", font=("Arial", 10)).grid(row=2, column=0, padx=5, pady=2, sticky="w")
    ttk.Label(stats_grid, text=f"{len(crossings)}").grid(row=2, column=1, padx=5, pady=2)

    ttk.Label(stats_grid, text="Total:", font=("Arial", 10, "bold")).grid(row=3, column=0, padx=5, pady=2, sticky="w")
    ttk.Label(stats_grid, text=f"{len(endings) + len(bifurcations) + len(crossings)}",
              font=("Arial", 10, "bold")).grid(row=3, column=1, padx=5, pady=2)

    # Button frame
    button_frame = ttk.Frame(window)
    button_frame.pack(fill="x", padx=10, pady=10)

    # Save button
    save_button = ttk.Button(
        button_frame,
        text='Save Results',
        command=lambda: save_results(binary, skeleton, highlighted, mask, minutiae)
    )
    save_button.pack(side="left", padx=5)

    # New analysis button
    new_button = ttk.Button(
        button_frame,
        text='New Analysis',
        command=reset_UI
    )
    new_button.pack(side="right", padx=5)


def save_results(binary, skeleton, highlighted, mask, minutiae):
    """Save the results to the timestamped folder"""
    try:
        # Create the timestamp folder if it doesn't exist
        if not os.path.exists(timestamped_folder_path):
            os.makedirs(timestamped_folder_path)

        # Save images
        cv2.imwrite(os.path.join(timestamped_folder_path, "binary.png"), binary)
        cv2.imwrite(os.path.join(timestamped_folder_path, "skeleton.png"), skeleton)
        cv2.imwrite(os.path.join(timestamped_folder_path, "minutiae.png"), highlighted)
        if mask is not None:
            cv2.imwrite(os.path.join(timestamped_folder_path, "mask.png"), mask)

        # Unpack minutiae
        endings, bifurcations, crossings = minutiae

        # Save minutiae data as CSV
        with open(os.path.join(timestamped_folder_path, "minutiae.csv"), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Type", "X", "Y"])

            for x, y in endings:
                writer.writerow(["Ending", x, y])

            for x, y in bifurcations:
                writer.writerow(["Bifurcation", x, y])

            for x, y in crossings:
                writer.writerow(["Crossing", x, y])

        # Save as figure
        plt.figure(figsize=(12, 8))

        plt.subplot(1, 3, 1)
        plt.imshow(binary, cmap='gray')
        plt.title("Binary")

        plt.subplot(1, 3, 2)
        plt.imshow(skeleton, cmap='gray')
        plt.title("Skeleton")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
        plt.title("Minutiae")

        plt.tight_layout()
        plt.savefig(os.path.join(timestamped_folder_path, "summary.png"))

        # Create a report
        with open(os.path.join(timestamped_folder_path, "report.txt"), "w") as f:
            f.write(f"Fingerprint Analysis Report\n")
            f.write(f"========================\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image: {image_name}\n\n")
            f.write(f"Minutiae Statistics:\n")
            f.write(f"- Endings: {len(endings)}\n")
            f.write(f"- Bifurcations: {len(bifurcations)}\n")
            f.write(f"- Crossings: {len(crossings)}\n")
            f.write(f"- Total: {len(endings) + len(bifurcations) + len(crossings)}\n")

        showinfo(title="Success", message=f"Results saved to {timestamped_folder_path}")
    except Exception as e:
        showinfo(title="Error", message=f"Failed to save results: {e}")


def kill_UI():
    """Remove all widgets from the window"""
    for widget in window.winfo_children():
        widget.destroy()


def reset_UI():
    """Reset the UI to initial state"""
    kill_UI()
    setup_initial_UI()


def setup_initial_UI():
    """Set up the initial UI"""
    # Frame for title and welcome message
    welcome_frame = ttk.Frame(window)
    welcome_frame.pack(fill="x", padx=20, pady=20)

    # Title and description
    ttk.Label(welcome_frame, text="Fingerprint Analysis System", font=("Arial", 16, "bold")).pack(pady=10)
    ttk.Label(welcome_frame, text="Upload fingerprint images for analysis and comparison").pack()

    # Frame for options
    options_frame = ttk.Frame(window)
    options_frame.pack(fill="both", expand=True, padx=20, pady=20)

    # Single analysis button
    single_button = ttk.Button(
        options_frame,
        text="Single Fingerprint Analysis",
        command=select_single_image,
        width=30
    )
    single_button.pack(pady=10)

    # Comparison button
    compare_button = ttk.Button(
        options_frame,
        text="Compare Two Fingerprints",
        command=select_two_images,
        width=30
    )
    compare_button.pack(pady=10)


def select_single_image():
    """Select a single image for analysis"""
    filename = fd.askopenfilename(
        title="Select fingerprint image",
        filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*"))
    )

    if filename:
        process_single_image(filename)


def select_two_images():
    """Select two images for comparison"""
    filename1 = fd.askopenfilename(
        title="Select first fingerprint image",
        filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*"))
    )

    if not filename1:
        return

    filename2 = fd.askopenfilename(
        title="Select second fingerprint image",
        filetypes=(("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*"))
    )

    if filename2:
        process_comparison(filename1, filename2)


def process_single_image(filename):
    """Process a single fingerprint image"""
    try:
        # Process path information
        global full_path, image_location, image_name, timestamped_folder_path
        image_location, image_name = os.path.split(filename)
        full_path = os.path.join(image_location, image_name).replace('\\', '/')

        # Create timestamp folder
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        timestamped_folder_path = os.path.join(image_location, timestamp)
        timestamped_folder_path = os.path.normpath(timestamped_folder_path).replace('\\', '/')

        # Create algorithm selection frame
        algo_frame = ttk.Frame(window)
        algo_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(algo_frame, text="Select Thinning Algorithm:").pack(side="left", padx=5)

        # Variable to store the selected algorithm
        algo_var = tk.StringVar(value="guo-hall")

        # Radio buttons for algorithm selection
        ttk.Radiobutton(
            algo_frame,
            text="Guo-Hall",
            variable=algo_var,
            value="guo-hall"
        ).pack(side="left", padx=10)

        ttk.Radiobutton(
            algo_frame,
            text="Zhang-Suen",
            variable=algo_var,
            value="zhang-suen"
        ).pack(side="left", padx=10)

        # Process button
        process_button = ttk.Button(
            algo_frame,
            text="Process Image",
            command=lambda: process_image(filename, algo_var.get())
        )
        process_button.pack(side="right", padx=10)

        # Back button
        back_button = ttk.Button(
            algo_frame,
            text="Back",
            command=reset_UI
        )
        back_button.pack(side="right", padx=10)

        # Display original image preview
        preview_frame = ttk.Frame(window)
        preview_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Load image
        cv_image = cv2.imread(filename)
        if cv_image is None:
            pil_image = Image.open(filename)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Process for preview
        if len(cv_image.shape) == 3:
            preview_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        else:
            preview_img = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

        # Resize and convert to Tkinter format
        pil_img = Image.fromarray(preview_img)
        pil_img.thumbnail((400, 400))
        tk_img = ImageTk.PhotoImage(pil_img)

        # Store reference to prevent garbage collection
        preview_frame.tk_img = tk_img

        # Display image
        img_label = ttk.Label(preview_frame, image=tk_img)
        img_label.pack(pady=10)
        ttk.Label(preview_frame, text=f"Image: {image_name}").pack()

    except Exception as e:
        showinfo(title="Error", message=f"{e}")
        reset_UI()


def process_image(filename, algo_name):
    """Process a single image with the selected algorithm"""
    try:
        # Set the algorithm
        if algo_name == "zhang-suen":
            algorithm = ThinningAlgorithm.ZHANG_SUEN
        else:
            algorithm = ThinningAlgorithm.GUO_HALL

        # Load image
        cv_image = cv2.imread(filename)
        if cv_image is None:
            pil_image = Image.open(filename)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Process the image
        binary, skeleton, highlighted, minutiae, mask = process_fingerprint(cv_image, algorithm)

        # Display results
        display_results(binary, skeleton, highlighted, minutiae, mask)

    except Exception as e:
        showinfo(title="Error", message=f"{e}")
        reset_UI()


def process_fingerprint(image, algorithm):
    """Process a fingerprint image to extract minutiae

    Args:
        image: The input image (color or grayscale)
        algorithm: The thinning algorithm to use

    Returns:
        binary: Binary image
        skeleton: Skeletonized image
        highlighted: Image with minutiae highlighted
        minutiae: Tuple of (endings, bifurcations, crossings) where each is a list of (x,y) coordinates
        mask: ROI mask or None if not used
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Create ROI mask (optional)
    mask = None

    # Perform preprocessing
    # Normalize
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Apply adaptive thresholding to get binary image
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Apply morphological operations to clean noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Skeletonize based on the selected algorithm
    if algorithm == ThinningAlgorithm.ZHANG_SUEN:
        skeleton = zhang_suen_thinning(binary)
    else:
        skeleton = guo_hall_thinning(binary)

    # Extract minutiae
    endings, bifurcations, crossings = extract_minutiae(skeleton)

    # Create highlighted image
    highlighted = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    # Mark endings with red
    for x, y in endings:
        cv2.circle(highlighted, (x, y), 5, (0, 0, 255), -1)

    # Mark bifurcations with green
    for x, y in bifurcations:
        cv2.circle(highlighted, (x, y), 5, (0, 255, 0), -1)

    # Mark crossings with blue
    for x, y in crossings:
        cv2.circle(highlighted, (x, y), 5, (255, 0, 0), -1)

    return binary, skeleton, highlighted, (endings, bifurcations, crossings), mask


def zhang_suen_thinning(binary_image):
    """Implement Zhang-Suen thinning algorithm"""
    skeleton = binary_image.copy() / 255  # Convert to 0 and 1

    changed = True
    while changed:
        changed = False

        # First sub-iteration
        marked = []
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                p2 = skeleton[i - 1, j]
                p3 = skeleton[i - 1, j + 1]
                p4 = skeleton[i, j + 1]
                p5 = skeleton[i + 1, j + 1]
                p6 = skeleton[i + 1, j]
                p7 = skeleton[i + 1, j - 1]
                p8 = skeleton[i, j - 1]
                p9 = skeleton[i - 1, j - 1]

                # Check if current pixel is 1
                if skeleton[i, j] == 1:
                    # Calculate A(P1) - number of 0->1 transitions
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                    transitions = sum((neighbors[k] == 0 and neighbors[k + 1] == 1) for k in range(8))

                    # Calculate B(P1) - number of non-zero neighbors
                    non_zero = np.sum([p2, p3, p4, p5, p6, p7, p8, p9])

                    # Check conditions
                    if (2 <= non_zero <= 6 and
                            transitions == 1 and
                            p2 * p4 * p6 == 0 and
                            p4 * p6 * p8 == 0):
                        marked.append((i, j))

        # Apply changes
        for i, j in marked:
            skeleton[i, j] = 0
            changed = True

        # Second sub-iteration
        marked = []
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                p2 = skeleton[i - 1, j]
                p3 = skeleton[i - 1, j + 1]
                p4 = skeleton[i, j + 1]
                p5 = skeleton[i + 1, j + 1]
                p6 = skeleton[i + 1, j]
                p7 = skeleton[i + 1, j - 1]
                p8 = skeleton[i, j - 1]
                p9 = skeleton[i - 1, j - 1]

                # Check if current pixel is 1
                if skeleton[i, j] == 1:
                    # Calculate A(P1) - number of 0->1 transitions
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
                    transitions = sum((neighbors[k] == 0 and neighbors[k + 1] == 1) for k in range(8))

                    # Calculate B(P1) - number of non-zero neighbors
                    non_zero = np.sum([p2, p3, p4, p5, p6, p7, p8, p9])

                    # Check conditions
                    if (2 <= non_zero <= 6 and
                            transitions == 1 and
                            p2 * p4 * p8 == 0 and
                            p2 * p6 * p8 == 0):
                        marked.append((i, j))

        # Apply changes
        for i, j in marked:
            skeleton[i, j] = 0
            changed = True

    return (skeleton * 255).astype(np.uint8)


def guo_hall_thinning(binary_image):
    """Implement Guo-Hall thinning algorithm"""
    skeleton = binary_image.copy() / 255  # Convert to 0 and 1

    changed = True
    while changed:
        changed = False

        # First sub-iteration
        marked = []
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if skeleton[i, j] != 1:
                    continue

                p2 = skeleton[i - 1, j]
                p3 = skeleton[i - 1, j + 1]
                p4 = skeleton[i, j + 1]
                p5 = skeleton[i + 1, j + 1]
                p6 = skeleton[i + 1, j]
                p7 = skeleton[i + 1, j - 1]
                p8 = skeleton[i, j - 1]
                p9 = skeleton[i - 1, j - 1]

                C = int((p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) +
                        (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) +
                        (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) +
                        (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1))

                N1 = int(p9 + p2 + p3 + p4 + p5 + p6 + p7 + p8)
                N2 = int(p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)

                if (C == 1 and
                        2 <= N1 <= 6 and
                        (p2 * p4 * p6 == 0) and
                        (p4 * p6 * p8 == 0)):
                    marked.append((i, j))

        # Apply changes
        for i, j in marked:
            skeleton[i, j] = 0
            changed = True

        # Second sub-iteration
        marked = []
        for i in range(1, skeleton.shape[0] - 1):
            for j in range(1, skeleton.shape[1] - 1):
                if skeleton[i, j] != 1:
                    continue

                p2 = skeleton[i - 1, j]
                p3 = skeleton[i - 1, j + 1]
                p4 = skeleton[i, j + 1]
                p5 = skeleton[i + 1, j + 1]
                p6 = skeleton[i + 1, j]
                p7 = skeleton[i + 1, j - 1]
                p8 = skeleton[i, j - 1]
                p9 = skeleton[i - 1, j - 1]

                C = int((p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) +
                        (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) +
                        (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) +
                        (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1))

                N1 = int(p9 + p2 + p3 + p4 + p5 + p6 + p7 + p8)
                N2 = int(p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)

                if (C == 1 and
                        2 <= N1 <= 6 and
                        (p2 * p4 * p8 == 0) and
                        (p2 * p6 * p8 == 0)):
                    marked.append((i, j))

        # Apply changes
        for i, j in marked:
            skeleton[i, j] = 0
            changed = True

    return (skeleton * 255).astype(np.uint8)


def extract_minutiae(skeleton):
    """Extract minutiae (endings, bifurcations, crossings) from a skeleton image
    Returns:
        endings: List of (x,y) coordinates of endings
        bifurcations: List of (x,y) coordinates of bifurcations
        crossings: List of (x,y) coordinates of crossings
    """
    endings = []
    bifurcations = []
    crossings = []

    # Pad the skeleton to handle edge cases
    padded = np.pad(skeleton, pad_width=1, mode='constant', constant_values=0)

    # Iterate through each pixel in the skeleton (except borders)
    for i in range(1, padded.shape[0] - 1):
        for j in range(1, padded.shape[1] - 1):
            # Check if the current pixel is part of the skeleton
            if padded[i, j] == 255:
                # Extract 3x3 neighborhood
                neighbors = padded[i - 1:i + 2, j - 1:j + 2].copy()
                center = neighbors[1, 1]
                neighbors[1, 1] = 0  # Remove center for counting

                # Count non-zero neighbors
                count = np.sum(neighbors == 255)

                # Classify minutiae based on neighbor count
                if count == 1:
                    endings.append((j - 1, i - 1))  # Convert to (x,y) format
                elif count == 3:
                    bifurcations.append((j - 1, i - 1))
                elif count >= 4:
                    crossings.append((j - 1, i - 1))

    return endings, bifurcations, crossings


class ThinningAlgorithm(Enum):
    """Enum for thinning algorithm selection"""
    ZHANG_SUEN = 1
    GUO_HALL = 2


# Create the main window
window = tk.Tk()
window.title("Fingerprint Analysis System")
window.geometry("1000x920")
window.resizable(False, False)

# Initialize global variables
full_path = None
filename = None
image_location = None
image_name = None
timestamped_folder_path = None

# Set up the initial UI
setup_initial_UI()

# Start the main loop
window.mainloop()