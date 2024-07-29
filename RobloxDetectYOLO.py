import cv2
import torch
import numpy as np
import mss
import time

# Load YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s').to(device)

# URL for the CCTV camera
camera_url = 'rtsp://admin:kowa1234@@192.168.1.100:554/unicast/c4/s0'

# Flag to switch between screen capture and CCTV camera
use_camera = False  # Set to True to use CCTV camera

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

# Function to capture frame from CCTV camera
def capture_camera():
    cap = cv2.VideoCapture(camera_url)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to capture from CCTV camera")
        return None
    return frame

# Function to run detection
def run_detection(frame):
    img = [frame]  # YOLOv5 expects a list of images
    results = model(img)
    return results

# Function to draw bounding boxes on frame
def draw_boxes(frame, results):
    for det in results.xyxy[0]:  # detections per image
        *xyxy, conf, cls = det
        label = f'{model.names[int(cls)]} {conf:.2f}'
        xyxy = [int(xy) for xy in xyxy]
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Main function
def main():
    resize_factor = 0.5  # Change this value to adjust the resize factor
    capture_interval = 1 / 30  # 30 FPS -> 1/30 seconds

    cap = None
    if use_camera:
        cap = cv2.VideoCapture(camera_url)

    while True:
        start_time = time.time()
        
        # Capture frame from screen or CCTV camera
        if use_camera and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture from CCTV camera")
                break
            if resize_factor != 1.0:
                height, width = frame.shape[:2]
                frame = cv2.resize(frame, (int(width * resize_factor), int(height * resize_factor)))
        else:
            frame = capture_screen(resize_factor)
        
        if frame is None:
            continue
        
        # Run detection
        results = run_detection(frame)
        
        # Draw bounding boxes
        frame = draw_boxes(frame, results)
        
        # Display frame
        cv2.imshow('YOLOv5 Object Detection', frame)
        
        # Wait for key press or end of capture interval
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        
        # Pause to maintain desired FPS
        elapsed_time = time.time() - start_time
        if elapsed_time < capture_interval:
            time.sleep(capture_interval - elapsed_time)

    if cap:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()