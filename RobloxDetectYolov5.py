import cv2
import torch
import numpy as np
import mss
import time

# Load YOLOv5 model with the path to the trained weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/ohmstylex/Desktop/AutoClick/yolov5/runs/train/exp/weights/best.pt')

# Set the model to evaluate mode
model.eval()

# Set the confidence threshold
model.conf = 0.25  # Minimum confidence to register a detection

# Function to capture and resize screen
def capture_screen(resize_factor=1.0):
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = np.array(sct.grab(monitor))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        if resize_factor != 1.0:
            height, width = screenshot.shape[:2]
            screenshot = cv2.resize(screenshot, (int(width * resize_factor), int(height * resize_factor)))
        return screenshot

# Function to run detection
def run_detection(frame):
    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model([frame_rgb])  # Pass the frame as a list
    return results

# Function to draw bounding boxes on frame
def draw_boxes(frame, results):
    for det in results.xyxy[0]:  # detections per image
        *xyxy, conf, cls = det
        if conf >= 0.11:  # Apply confidence filter
            label = f'{model.names[int(cls)]} {conf:.2f}'
            xyxy = [int(xy) for xy in xyxy]
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Main function for real-time detection using screen capture
def main():
    resize_factor = 0.5  # Adjust this value to resize the captured screen
    capture_interval = 1 / 60  # Set desired FPS (e.g., 30 FPS)

    while True:
        start_time = time.time()
        
        # Capture and resize screen
        frame = capture_screen(resize_factor)
        
        # Run detection
        results = run_detection(frame)
        
        # Draw bounding boxes
        frame = draw_boxes(frame, results)
        
        # Display frame
        cv2.imshow('YOLOv5 Object Detection', frame)
        
        # Wait for key press or end of capture interval
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Pause to maintain desired FPS
        elapsed_time = time.time() - start_time
        if elapsed_time < capture_interval:
            time.sleep(capture_interval - elapsed_time)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
