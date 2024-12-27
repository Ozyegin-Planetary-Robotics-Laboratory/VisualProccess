import os
from ultralytics import YOLO
import cv2

# Dynamically find the path to 'runs/detect/train13/weights/best.pt'
current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
model_path = os.path.join(current_dir, '..', 'runs', 'detect', 'train13', 'weights', 'best.pt')
model_path = os.path.normpath(model_path)  # Normalize the path for different OS

# Load trained YOLOv8 model
model = YOLO(model_path)

# Open a webcam feed (or use an external camera)
cap = cv2.VideoCapture(0)  # Replace 0 with the camera index or video file path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model.predict(source=frame, show=False, conf=0.5)  # Adjust confidence threshold

    # Draw bounding boxes and labels on the frame
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, confidence, cls = box
            label = f"{model.names[int(cls)]} {confidence:.2f}"

            # Draw a rectangle and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Orange Hammer Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
