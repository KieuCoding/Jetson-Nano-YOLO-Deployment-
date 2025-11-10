import cv2
import os
from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO('yolov8n.pt')  # Ensure best.pt is in the current directory

# Create directories if they don't exist
os.makedirs('captured_images', exist_ok=True)
os.makedirs('trash_pictures', exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Image counter
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run inference on the frame
    results = model(frame, imgsz=640, conf=0.3)  # Adjust conf if needed

    # Annotate the frame with results
    annotated_frame = results[0].plot()

    # Show the annotated frame
    cv2.imshow('YOLOv8 Live Inference', annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        # Save image
        img_name = f'captured_images/image_{count}.png'
        cv2.imwrite(img_name, frame)
        print(f'Image saved as {img_name}')
        count += 1

    elif key == ord('d'):
        # Discard image to trash
        img_name = f'trash_pictures/image_{count}.png'
        cv2.imwrite(img_name, frame)
        print(f'Image discarded as {img_name}')
        count += 1

    elif key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
