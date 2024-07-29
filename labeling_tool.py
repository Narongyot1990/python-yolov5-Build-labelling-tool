import cv2
import os
import yaml
import shutil
import random

# Directory paths
images_dir = 'labelingImg/images'  # Update this path to your images directory
labels_dir = 'labelingImg/labels'  # Update this path to your labels directory
dataset_dir = 'labelingImg/dataset'  # Directory for the dataset

# Create the labels directory if it doesn't exist
os.makedirs(labels_dir, exist_ok=True)

# Global variables
current_img = None
current_img_copy = None
bboxes = []
undo_stack = []
redo_stack = []
current_class = ''
class_names = []
dragging = False
drawing = False
moving = False
resizing = False
selected_bbox_index = None
selected_class_index = None
crosshair = None
current_image_index = 0
image_paths = []
offset_x = 0
offset_y = 0

# Load image paths
def load_image_paths():
    global image_paths
    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_paths.sort()

# Mouse callback function
def draw_bbox(event, x, y, flags, param):
    global bboxes, current_img, current_img_copy, dragging, drawing, moving, resizing, crosshair, selected_bbox_index, selected_class_index, offset_x, offset_y
    crosshair = (x, y)
    margin = 10  # Margin for edge selection
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing:
            bboxes.append([[x, y], [x, y], current_class])
            dragging = True
        else:
            selected_bbox_index = None
            selected_class_index = None
            for i, (pt1, pt2, cls) in enumerate(bboxes):
                if (abs(x - pt1[0]) <= margin or abs(x - pt2[0]) <= margin or abs(y - pt1[1]) <= margin or abs(y - pt2[1]) <= margin):
                    selected_bbox_index = i
                    resizing = True
                    break
                elif pt1[0] < x < pt2[0] and pt1[1] < y < pt2[1]:
                    selected_bbox_index = i
                    moving = True
                    offset_x = x - pt1[0]
                    offset_y = y - pt1[1]
                    break
                if pt1[0] <= x <= pt1[0] + 150 and pt1[1] - 30 <= y <= pt1[1]:
                    selected_class_index = i
                    break
            update_image()
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            bboxes[-1][1] = [x, y]
        elif resizing and selected_bbox_index is not None:
            pt1, pt2, cls = bboxes[selected_bbox_index]
            if abs(x - pt1[0]) <= margin:
                pt1[0] = x
            elif abs(x - pt2[0]) <= margin:
                pt2[0] = x
            if abs(y - pt1[1]) <= margin:
                pt1[1] = y
            elif abs(y - pt2[1]) <= margin:
                pt2[1] = y
            bboxes[selected_bbox_index] = [pt1, pt2, cls]
        elif moving and selected_bbox_index is not None:
            pt1, pt2, cls = bboxes[selected_bbox_index]
            pt1[0] = x - offset_x
            pt1[1] = y - offset_y
            pt2[0] = pt1[0] + (pt2[0] - pt1[0])
            pt2[1] = pt1[1] + (pt2[1] - pt1[1])
            bboxes[selected_bbox_index] = [pt1, pt2, cls]
        update_image()
    
    elif event == cv2.EVENT_LBUTTONUP:
        if dragging:
            dragging = False
            bboxes[-1][1] = [x, y]
        elif resizing:
            resizing = False
        elif moving:
            moving = False
        update_image()

# Function to draw crosshair lines
def draw_crosshair(image, x, y):
    height, width = image.shape[:2]
    cv2.line(image, (x, 0), (x, height), (255, 0, 0), 1)
    cv2.line(image, (0, y), (width, y), (255, 0, 0), 1)

# Function to save annotation
def save_annotation(image_path, bboxes):
    global class_names
    image_name = os.path.basename(image_path)
    label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')
    h, w, _ = current_img.shape
    with open(label_path, 'w') as f:
        for bbox, cls in [(bbox[:2], bbox[2]) for bbox in bboxes]:
            pt1, pt2 = bbox
            x_center = (pt1[0] + pt2[0]) / 2 / w
            y_center = (pt1[1] + pt2[1]) / 2 / h
            width = abs(pt2[0] - pt1[0]) / w
            height = abs(pt2[1] - pt1[1]) / h
            class_index = class_names.index(cls)
            f.write(f'{class_index} {x_center} {y_center} {width} {height}\n')

# Function to update image display
def update_image():
    global current_img_copy
    current_img_copy = current_img.copy()
    for i, (bbox, cls) in enumerate([(bbox[:2], bbox[2]) for bbox in bboxes]):
        pt1, pt2 = bbox
        color = (0, 255, 0) if i != selected_bbox_index else (0, 0, 255)
        cv2.rectangle(current_img_copy, tuple(pt1), tuple(pt2), color, 2)
        class_color = (0, 255, 0) if i != selected_class_index else (0, 0, 255)
        cv2.putText(current_img_copy, cls, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, class_color, 2, cv2.LINE_AA)
    if crosshair:
        draw_crosshair(current_img_copy, crosshair[0], crosshair[1])
    cv2.imshow('image', current_img_copy)

# Delete the selected bounding box
def delete_bbox():
    global selected_bbox_index
    if selected_bbox_index is not None:
        undo_stack.append(bboxes.pop(selected_bbox_index))
        selected_bbox_index = None
        update_image()

# Set class name for the selected class
def set_class_name():
    global selected_class_index
    if selected_class_index is not None:
        new_class = input("Enter new class name: ")
        if new_class:
            undo_stack.append((selected_class_index, bboxes[selected_class_index][2]))
            bboxes[selected_class_index][2] = new_class
            if new_class not in class_names:
                class_names.append(new_class)
        selected_class_index = None
        update_image()

# Change default class name
def change_default_class_name():
    global current_class
    new_class = input("Enter new default class name: ")
    if new_class:
        current_class = new_class
        if new_class not in class_names:
            class_names.append(new_class)
    print(f"Default class name set to: {current_class}")

# Undo the last action
def undo():
    if undo_stack:
        action = undo_stack.pop()
        if isinstance(action, list):
            bboxes.append(action)
        elif isinstance(action, tuple):
            index, prev_class = action
            bboxes[index][2] = prev_class
        redo_stack.append(action)
        update_image()

# Redo the last undone action
def redo():
    if redo_stack:
        action = redo_stack.pop()
        if isinstance(action, list):
            bboxes.pop()
        elif isinstance(action, tuple):
            index, prev_class = action
            bboxes[index][2] = prev_class
        undo_stack.append(action)
        update_image()

# Load the current image
def load_image(index):
    global current_img, current_img_copy, bboxes, crosshair, current_image_index, undo_stack, redo_stack
    if 0 <= index < len(image_paths):
        current_image_index = index
        image_path = image_paths[current_image_index]
        current_img = cv2.imread(image_path)
        current_img_copy = current_img.copy()
        bboxes = []
        crosshair = None
        undo_stack = []
        redo_stack = []
        cv2.imshow('image', current_img_copy)
        cv2.setMouseCallback('image', draw_bbox)
        load_annotations(image_path)
        print(f"Annotating {os.path.basename(image_path)} ({current_image_index + 1}/{len(image_paths)}). Press 'n' for next image, 'p' for previous image, 'd' to delete selected bounding box, 's' to set class name, 'c' to change default class name, 'u' to undo, 'r' to redo, 't' to toggle drawing, 'q' to quit, 'y' to create data.yaml and distribute images.")
        update_image()

# Load annotations if available
def load_annotations(image_path):
    global bboxes
    bboxes = []
    label_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(image_path))[0] + '.txt')
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            h, w, _ = current_img.shape
            for line in lines:
                data = line.strip().split()
                try:
                    cls_index = int(data[0])
                    cls = class_names[cls_index]
                    x_center, y_center, width, height = map(float, data[1:])
                    x_center *= w
                    y_center *= h
                    width *= w
                    height *= h
                    pt1 = [int(x_center - width / 2), int(y_center - height / 2)]
                    pt2 = [int(x_center + width / 2), int(y_center + height / 2)]
                    bboxes.append([pt1, pt2, cls])
                except ValueError:
                    print(f"Skipping invalid annotation: {line.strip()}")

# Toggle drawing mode
def toggle_drawing():
    global drawing, crosshair
    drawing = not drawing
    crosshair = None if not drawing else (0, 0)
    print(f"Drawing mode {'enabled' if drawing else 'disabled'}")
    update_image()

# Prompt for initial class name
def prompt_class_name():
    global current_class, class_names
    current_class = input("Enter default class name: ")
    if current_class:
        if current_class not in class_names:
            class_names.append(current_class)

# Create dataset folder template
def create_dataset_template():
    for folder in ['train/images', 'train/labels', 'val/images', 'val/labels']:
        os.makedirs(os.path.join(dataset_dir, folder), exist_ok=True)
    print(f"Dataset template created at {dataset_dir}")

# Create data.yaml file
def create_data_yaml():
    global class_names
    data_yaml = {
        'train': os.path.join(dataset_dir, 'train/images'),
        'val': os.path.join(dataset_dir, 'val/images'),
        'nc': len(class_names),
        'names': class_names
    }
    with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data_yaml, f)
    print(f"data.yaml created at {os.path.join(dataset_dir, 'data.yaml')}")

# Distribute images to train and validation sets
def distribute_images(train_percentage):
    image_paths = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_paths)
    train_count = int(len(image_paths) * (train_percentage / 100))
    
    train_images = image_paths[:train_count]
    val_images = image_paths[train_count:]
    
    def copy_files(file_paths, target_folder):
        for file_path in file_paths:
            image_name = os.path.basename(file_path)
            label_name = os.path.splitext(image_name)[0] + '.txt'
            image_target_dir = os.path.join(target_folder, 'images')
            label_target_dir = os.path.join(target_folder, 'labels')
            os.makedirs(image_target_dir, exist_ok=True)
            os.makedirs(label_target_dir, exist_ok=True)
            shutil.copy(file_path, os.path.join(image_target_dir, image_name))
            label_path = os.path.join(labels_dir, label_name)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(label_target_dir, label_name))

    copy_files(train_images, os.path.join(dataset_dir, 'train'))
    copy_files(val_images, os.path.join(dataset_dir, 'val'))
    print(f"Distributed {len(train_images)} images to train and {len(val_images)} images to val")

# Main function
def main():
    global current_image_index
    prompt_class_name()
    load_image_paths()
    if not image_paths:
        print("No images found in the specified directory.")
        return

    create_dataset_template()
    
    load_image(current_image_index)
    
    while True:
        key = cv2.waitKey(1)
        if key == ord('n'):
            save_annotation(image_paths[current_image_index], bboxes)
            load_image((current_image_index + 1) % len(image_paths))  # Loop to the first image
        elif key == ord('p'):
            save_annotation(image_paths[current_image_index], bboxes)
            load_image((current_image_index - 1) % len(image_paths))  # Loop to the last image
        elif key == ord('d'):
            delete_bbox()
        elif key == ord('s'):
            set_class_name()
        elif key == ord('c'):
            change_default_class_name()
        elif key == ord('u'):
            undo()
        elif key == ord('r'):
            redo()
        elif key == ord('t'):
            toggle_drawing()
        elif key == ord('y'):
            create_data_yaml()
            train_percentage = float(input("Enter the percentage of images for training (0-100): "))
            distribute_images(train_percentage)
        elif key == ord('q'):
            save_annotation(image_paths[current_image_index], bboxes)
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
