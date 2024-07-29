import cv2
import torch
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/custom_yolov5s_results8/weights/best.pt')

# Function to run detection
def run_detection(frame):
    results = model(frame)
    return results

# Function to draw bounding boxes on frame
def draw_boxes(frame, results, conf_threshold=0.25):
    for det in results.xyxy[0]:  # detections per image
        *xyxy, conf, cls = det
        if conf > conf_threshold:  # Only draw boxes above the confidence threshold
            label = f'{model.names[int(cls)]} {conf:.2f}'
            xyxy = [int(xy) for xy in xyxy]
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Main function to test with a static image
def main():
    image_path = '/Users/ohmstylex/Desktop/AutoClick/yolov5/test_image.jpg'  # Update this path to your test image

    # Check if the image file exists
    if not os.path.isfile(image_path):
        print(f"Image file not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image file: {image_path}")
        return

    results = run_detection(frame)
    print("Detection Results:", results.pandas().xyxy[0])  # Print the results for debugging

    frame = draw_boxes(frame, results)
    cv2.imshow('YOLOv5 Object Detection', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
