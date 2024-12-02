import cv2
import time
# Load a video
video_path = "clip.mp4"
cap = cv2.VideoCapture(video_path)

# Read two consecutive frames
ret, frame1 = cap.read()
for i in range(0, 5):
    cap.read()
ret, frame2 = cap.read()

# Save the frames as images
cv2.imwrite("frame1.png", frame1)
cv2.imwrite("frame2.png", frame2)

cap.release()
