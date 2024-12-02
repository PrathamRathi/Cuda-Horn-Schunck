import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_derivatives(I1, I2):
    """
    Compute spatial and temporal derivatives between two images.
    """
    # Spatial derivatives using central difference
    Ix = (np.roll(I1, -1, axis=1) - np.roll(I1, 1, axis=1)) / 2.0
    Iy = (np.roll(I1, -1, axis=0) - np.roll(I1, 1, axis=0)) / 2.0

    # Temporal derivative using forward difference
    It = I2 - I1

    return Ix, Iy, It

def horn_schunk(I1, I2, alpha=1.0, num_iterations=100):
    """
    Perform Horn-Schunck optical flow computation.
    Args:
        I1: First image (grayscale, numpy array)
        I2: Second image (grayscale, numpy array)
        alpha: Smoothness regularization parameter
        num_iterations: Number of iterations
    Returns:
        u: Horizontal optical flow
        v: Vertical optical flow
    """
    # Compute image derivatives
    Ix, Iy, It = compute_derivatives(I1, I2)

    # Initialize flow fields
    u = np.zeros_like(I1)
    v = np.zeros_like(I1)

    # Iterative update
    for _ in range(num_iterations):
        # Compute local averages of the flow fields
        u_avg = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                 np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)) / 4.0
        v_avg = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
                 np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1)) / 4.0

        # Update flow fields
        numerator = (Ix * u_avg + Iy * v_avg + It)
        denominator = alpha**2 + Ix**2 + Iy**2
        u = u_avg - Ix * numerator / denominator
        v = v_avg - Iy * numerator / denominator

    return u, v

def draw_flow_on_image(image, u, v, step=10, color=(0, 255, 0)):
    """
    Draw optical flow as arrows on top of an image.
    Args:
        image: Background image (grayscale or RGB)
        u: Horizontal flow
        v: Vertical flow
        step: Sampling step for arrows
        color: Arrow color (BGR for OpenCV)
    Returns:
        overlay: Image with optical flow arrows
    """
    if len(image.shape) == 2:  # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    overlay = image.copy()
    height, width = u.shape

    # Sample flow for visualization
    for y in range(0, height, step):
        for x in range(0, width, step):
            start_point = (x, y)
            end_point = (int(x + u[y, x]), int(y + v[y, x]))
            cv2.arrowedLine(overlay, start_point, end_point, color, 1, tipLength=0.3)

    return overlay

# Main execution
if __name__ == "__main__":
    # Load two grayscale images
    I1 = cv2.imread("car1.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    I2 = cv2.imread("car2.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # Run Horn-Schunck optical flow
    u, v = horn_schunk(I1, I2, alpha=1.0, num_iterations=100)

    # Overlay flow arrows on the first image
    result_image = draw_flow_on_image(cv2.imread("car1.jpg"), u, v, step=10, color=(0, 255, 0))

    # Display the result
    cv2.imshow("Optical Flow Arrows", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result if needed
    cv2.imwrite("optical_flow_result.png", result_image)
