import numpy as np
import cv2

def horn_schunck_optical_flow(I1, I2, alpha=1.0, max_iter=100, epsilon=1e-6):
    """
    Implement Horn-Schunck optical flow algorithm.
    
    Parameters:
    -----------
    I1 : numpy.ndarray
        First input image (grayscale)
    I2 : numpy.ndarray
        Second input image (grayscale)
    alpha : float, optional
        Regularization parameter (default: 1.0)
        Controls the smoothness of the flow field
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    epsilon : float, optional
        Convergence threshold (default: 1e-6)
    
    Returns:
    --------
    u : numpy.ndarray
        Horizontal component of optical flow
    v : numpy.ndarray
        Vertical component of optical flow
    """
    # Ensure input images are grayscale and of the same size
    assert I1.shape == I2.shape, "Input images must have the same dimensions"
    assert len(I1.shape) == 2, "Input images must be grayscale"
    
    # Image dimensions
    height, width = I1.shape
    
    # Initialize flow components
    u = np.zeros_like(I1, dtype=np.float32)
    v = np.zeros_like(I1, dtype=np.float32)
    
    # Compute image gradients
    fx = cv2.Sobel(I1, cv2.CV_32F, 1, 0, ksize=3)
    fy = cv2.Sobel(I1, cv2.CV_32F, 0, 1, ksize=3)
    ft = I2 - I1
    
    # Iterative optimization
    for _ in range(max_iter):
        # Compute local averages of flow components
        u_avg = cv2.boxFilter(u, cv2.CV_32F, (3, 3)) 
        v_avg = cv2.boxFilter(v, cv2.CV_32F, (3, 3))
        
        # Compute flow update
        numerator = (
            fx * u_avg + 
            fy * v_avg + 
            fx * ft
        )
        denominator = (
            alpha**2 + 
            fx**2 + 
            fy**2
        )
        
        # Updated flow components
        u_new = u_avg - fx * (numerator / denominator)
        v_new = v_avg - fy * (numerator / denominator)
        
        # Check convergence
        diff = np.max(np.abs(u_new - u)) + np.max(np.abs(v_new - v))
        
        # Update flow components
        u = u_new
        v = v_new
        
        # Break if converged
        if diff < epsilon:
            break
    
    return u, v

def visualize_optical_flow(image, u, v, scale=10):
    """
    Visualize optical flow vectors on the original image.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original image
    u : numpy.ndarray
        Horizontal flow component
    v : numpy.ndarray
        Vertical flow component
    scale : float, optional
        Scale factor for flow vectors (default: 10)
    
    Returns:
    --------
    numpy.ndarray
        Image with optical flow vectors drawn
    """
    # Create a color image from grayscale if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Create a copy to draw on
    flow_image = image.copy()
    
    # Sample points (to avoid overcrowding)
    step = 10
    for y in range(0, u.shape[0], step):
        for x in range(0, u.shape[1], step):
            # Get flow vector
            dx = u[y, x] * scale
            dy = v[y, x] * scale
            
            # Draw flow vector
            cv2.arrowedLine(
                flow_image, 
                (x, y), 
                (int(x + dx), int(y + dy)), 
                (0, 255, 0),  # Green color
                thickness=1,
                tipLength=0.3
            )
    
    return flow_image

# Example usage
def main():
    # Load two consecutive frames
    frame1 = cv2.imread('frame1.png', cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread('frame2.png', cv2.IMREAD_GRAYSCALE)
    
    # Compute optical flow
    u, v = horn_schunck_optical_flow(frame1, frame2, alpha=1.0)
    
    # Visualize flow
    flow_vis = visualize_optical_flow(frame1, u, v)
    
    # Display results
    cv2.imshow('Optical Flow', flow_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()