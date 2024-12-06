import cv2

def resize_image(image_path, resize_sizes):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image.")
        return
    
    print(f"Original image dimensions: {image.shape[1]}x{image.shape[0]}")
    
    for idx, (width, height) in enumerate(resize_sizes):
        # Resize the image
        resized = cv2.resize(image, (width, height))
        
        # Save the resized image
        output_path = f"car{width}f2.jpg"
        cv2.imwrite(output_path, resized)
        print(f"Resized image saved as '{output_path}'.")

# Input: Path to the image and a list of resize dimensions
image_path = "images/car2.jpg"  # Replace with your image path
resize_sizes = [(512, 512), (256, 256), (128, 128)]  # List of (width, height) dimensions

resize_image(image_path, resize_sizes)
