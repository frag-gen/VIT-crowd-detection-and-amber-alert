<<<<<<< HEAD
import threading
import cv2
import numpy as np
import os
import requests
import time
from datetime import datetime
from flask import Flask, Response, jsonify

# Thread-safe last_alert_time dictionary
last_alert_time = {}
alert_time_lock = threading.Lock()

# Initialize Flask app
app = Flask(__name__)

# Load VIT campus map
file_path = r"C:\Users\aksha\OneDrive\Desktop\VIT-MAP.png"

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

map_image = cv2.imread(file_path)
if map_image is None:
    print("Error: Unable to load the map image. Check the file path and format.")
    exit()

# Define the single camera location (Canteen)
camera_location = {
    "name": "Canteen",
    "coords": (600, 400),  # Adjust based on your map
    "density": "green"
}

# Define density colors with transparency
density_colors = {
    "green": (0, 255, 0, 128),    # Low density
    "orange": (0, 165, 255, 128), # Medium density
    "red": (0, 0, 255, 128)       # High density
}

# Load class labels for COCO dataset
def load_classes(file_path):
    with open(file_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Load YOLO model
weights_path = r"C:\Users\aksha\Downloads\yolov4.weights"
config_path = r"C:\Users\aksha\Downloads\yolov4.cfg"
coco_names_path = r"C:\Users\aksha\Downloads\coco.names"

# Check if the YOLO files exist
if not os.path.exists(weights_path) or not os.path.exists(config_path) or not os.path.exists(coco_names_path):
    print("YOLO files not found. Please ensure yolov4.weights, yolov4.cfg, and coco.names are in the correct directory.")
    exit()

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()

# Check the shape of the output layers
unconnected_out_layers = net.getUnconnectedOutLayers()

# If unconnected_out_layers is a single-dimensional array, use it directly
if len(unconnected_out_layers.shape) == 1:
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
else:
    # If it's two-dimensional, flatten it first
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]

classes = load_classes(coco_names_path)

# Threading for video smoothening
frame_lock = threading.Lock()
frame = None

# Function to send an alert with a 20-minute check
def send_alert_to_backend(people_count, camera_name):
    global last_alert_time

    current_time = datetime.now()

    # Locking the access to last_alert_time to avoid race conditions
    with alert_time_lock:
        print(f"Checking last alert time for {camera_name}")
        
        # Check if an alert was sent within the last 20 minutes for the same location
        if camera_name in last_alert_time:
            last_alert = last_alert_time[camera_name]
            print(f"Last alert time for {camera_name}: {last_alert}")
            time_diff = current_time - last_alert
            print(f"Time since last alert: {time_diff.total_seconds()} seconds")
            
            # If less than 20 minutes, skip sending
            if time_diff.total_seconds() < 1200:  # 1200 seconds = 20 minutes
                print(f"Alert for {camera_name} skipped. Already sent within the last 20 minutes.")
                return  # Skip sending the alert

        # Your Flask API endpoint for alerting
        url = "http://127.0.0.1:5000/send_alert"  # Replace with your Flask server URL

        payload = {
            'cameraName': camera_name,            # Use 'cameraName' instead of 'camera_name'
            'peopleCount': people_count,          # Use 'peopleCount' instead of 'people_count'
            'status': 'red' if people_count >= 5 else 'green',  # Crowd density based on count
        }

        # Send alert to Flask backend (Local server)
        try:
            response = requests.post(url, json=payload)  # Use json instead of data
            print(f"Sent alert to Flask backend for {camera_name}, Response: {response.status_code}")
            
            # Update last_alert_time only after the alert is sent successfully
            last_alert_time[camera_name] = current_time  # Update last alert time after sending
            print(f"Last alert time for {camera_name} updated to: {current_time}")
        except Exception as e:
            print(f"Error sending alert to Flask backend: {e}")

def capture_frames():
    global frame
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Force DirectShow as backend
    while True:
        ret, new_frame = camera.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        with frame_lock:
            frame = new_frame

# Start video capture in a separate thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True
capture_thread.start()

# Function to detect crowd density and draw overlays
def detect_crowd_density(frame, net, output_layers, classes):
    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 480))

    # Prepare the input blob for YOLO
    blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # Use 416x416 for YOLOv4
    net.setInput(blob)
    outs = net.forward(output_layers)

    height, width, _ = frame_resized.shape
    bboxes = []
    confidences = []
    class_ids = []
    people_count = 0

    # Loop through YOLO's outputs to collect detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter detections for "person" class with high confidence
            if confidence > 0.5 and classes[class_id] == "person":
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bboxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.4)  # Adjust thresholds if needed
    if len(indices) > 0:  # Check if there are valid indices
        for i in indices.flatten():  # Use .flatten() to iterate directly
            x, y, w, h = bboxes[i]
            
            # Draw bounding box on the frame
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            people_count += 1

            # Display the confidence score
            cv2.putText(frame_resized, f"Person {confidences[i]:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Determine crowd density based on the number of people detected
    if people_count < 2:
        density_color = density_colors["green"]
    elif 2 <= people_count < 3:
        density_color = density_colors["orange"]
    else:
        density_color = density_colors["red"]

    # Add transparency and draw density on the map
    overlay = map_image.copy()
    radius = 50  # Increase the radius for the circle
    cv2.circle(overlay, camera_location["coords"], radius, density_color[:3], -1)

    # Add transparency
    alpha = density_color[3] / 255.0
    map_with_overlay = cv2.addWeighted(overlay, alpha, map_image, 1 - alpha, 0)

    # Draw density text in bold and black
    cv2.putText(map_with_overlay, f"Density: {people_count}", 
                (camera_location["coords"][0] - 30, camera_location["coords"][1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Black color, thickness 2

    # If red density detected, send the alert to Flask
    if density_color == density_colors["red"]:
        send_alert_to_backend(people_count, "Canteen")

    return frame_resized, map_with_overlay

# Video stream generator function (MJPEG stream)
def gen_frames():
    while True:
        if frame is not None:
            with frame_lock:
                processed_frame, updated_map, people_count = detect_crowd_density(frame)

                # Encode processed frame as JPEG for MJPEG streaming
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_data = buffer.tobytes()

                # Yield the frame in MJPEG format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

# Serve the MJPEG stream to the frontend
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to serve heatmap image (map overlay)
@app.route('/heatmap')
def heatmap():
    if frame is not None:
        _, buffer = cv2.imencode('.jpg', map_image)  # Encode the heatmap as JPEG
        heatmap_data = buffer.tobytes()
        return Response(heatmap_data, mimetype='image/jpeg')
    return "No image available", 404

# Route to serve index page (HTML page)
@app.route('/')
def index():
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
=======
import threading
import cv2
import numpy as np
import os
import requests
import time
from datetime import datetime

# Thread-safe last_alert_time dictionary
last_alert_time = {}
alert_time_lock = threading.Lock()

# Load VIT campus map
file_path = r"C:\Users\aksha\OneDrive\Desktop\VIT-MAP.png"

# Check if the file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

map_image = cv2.imread(file_path)
if map_image is None:
    print("Error: Unable to load the map image. Check the file path and format.")
    exit()

# Define the single camera location (Canteen)
camera_location = {
    "name": "Canteen",
    "coords": (600, 400),  # Adjust based on your map
    "density": "green"
}

# Define density colors with transparency
density_colors = {
    "green": (0, 255, 0, 128),    # Low density
    "orange": (0, 165, 255, 128), # Medium density
    "red": (0, 0, 255, 128)       # High density
}

# Load class labels for COCO dataset
def load_classes(file_path):
    with open(file_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Load YOLO model
weights_path = r"C:\Users\aksha\Downloads\yolov4.weights"
config_path = r"C:\Users\aksha\Downloads\yolov4.cfg"
coco_names_path = r"C:\Users\aksha\Downloads\coco.names"

# Check if the YOLO files exist
if not os.path.exists(weights_path) or not os.path.exists(config_path) or not os.path.exists(coco_names_path):
    print("YOLO files not found. Please ensure yolov4.weights, yolov4.cfg, and coco.names are in the correct directory.")
    exit()

# Load YOLO
net = cv2.dnn.readNet(weights_path, config_path)
layer_names = net.getLayerNames()

# Check the shape of the output layers
unconnected_out_layers = net.getUnconnectedOutLayers()

# If unconnected_out_layers is a single-dimensional array, use it directly
if len(unconnected_out_layers.shape) == 1:
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
else:
    # If it's two-dimensional, flatten it first
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]

classes = load_classes(coco_names_path)

# Threading for video smoothening
frame_lock = threading.Lock()
frame = None

# Function to send an alert with a 20-minute check
def send_alert_to_twilio(people_count, camera_name):
    global last_alert_time

    current_time = datetime.now()

    # Locking the access to last_alert_time to avoid race conditions
    with alert_time_lock:
        print(f"Checking last alert time for {camera_name}")
        
        # Check if an alert was sent within the last 20 minutes for the same location
        if camera_name in last_alert_time:
            last_alert = last_alert_time[camera_name]
            print(f"Last alert time for {camera_name}: {last_alert}")
            time_diff = current_time - last_alert
            print(f"Time since last alert: {time_diff.total_seconds()} seconds")
            
            # If less than 20 minutes, skip sending
            if time_diff.total_seconds() < 1200:  # 1200 seconds = 20 minutes
                print(f"Alert for {camera_name} skipped. Already sent within the last 20 minutes.")
                return  # Skip sending the alert

        # Your Twilio API sending logic goes here
        url = "http://127.0.0.1:5000/send_alert"  # Assuming you're using the Flask server

        payload = {
            'camera_name': camera_name,
            'people_count': people_count,
            'crowd_density': 'red' if people_count >= 5 else 'green',  # Example logic for crowd density
        }

        # Send alert logic (e.g., via Twilio API)
        try:
            response = requests.post(url, data=payload)
            print(f"Sent alert to Twilio for {camera_name}, Response: {response.status_code}")
            
            # Update last_alert_time only after the alert is sent successfully
            last_alert_time[camera_name] = current_time  # Update last alert time after sending
            print(f"Last alert time for {camera_name} updated to: {current_time}")
        except Exception as e:
            print(f"Error sending alert: {e}")

def capture_frames():
    global frame
    camera = cv2.VideoCapture(0)  # Use a camera or video file
    while True:
        ret, new_frame = camera.read()
        if ret:
            with frame_lock:
                frame = new_frame

# Start video capture in a separate thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.daemon = True
capture_thread.start()

# Function to detect crowd density and draw overlays
def detect_crowd_density(frame, net, output_layers, classes):
    # Resize frame for faster processing
    frame_resized = cv2.resize(frame, (640, 480))

    # Prepare the input blob for YOLO
    blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # Use 416x416 for YOLOv4
    net.setInput(blob)
    outs = net.forward(output_layers)

    height, width, _ = frame_resized.shape
    bboxes = []
    confidences = []
    class_ids = []
    people_count = 0

    # Loop through YOLO's outputs to collect detected objects
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter detections for "person" class with high confidence
            if confidence > 0.5 and classes[class_id] == "person":
                # Get bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bboxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.4)  # Adjust thresholds if needed
    if len(indices) > 0:  # Check if there are valid indices
        for i in indices.flatten():  # Use .flatten() to iterate directly
            x, y, w, h = bboxes[i]
            
            # Draw bounding box on the frame
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            people_count += 1

            # Display the confidence score
            cv2.putText(frame_resized, f"Person {confidences[i]:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Determine crowd density based on the number of people detected
    if people_count < 2:
        density_color = density_colors["green"]
    elif 2 <= people_count < 3:
        density_color = density_colors["orange"]
    else:
        density_color = density_colors["red"]

    # Add transparency and draw density on the map
    overlay = map_image.copy()
    radius = 50  # Increase the radius for the circle
    cv2.circle(overlay, camera_location["coords"], radius, density_color[:3], -1)

    # Add transparency
    alpha = density_color[3] / 255.0
    map_with_overlay = cv2.addWeighted(overlay, alpha, map_image, 1 - alpha, 0)

    # Draw density text in bold and black
    cv2.putText(map_with_overlay, f"Density: {people_count}", 
                (camera_location["coords"][0] - 30, camera_location["coords"][1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Black color, thickness 2

    # If red density detected, send the alert (to Code 2)
    if density_color == density_colors["red"]:
        send_alert_to_twilio(people_count, "Canteen")

    return frame_resized, map_with_overlay

# Main loop to process frames
while True:
    if frame is not None:
        processed_frame, updated_map = detect_crowd_density(frame, net, output_layers, classes)
        cv2.imshow("Crowd Detection", processed_frame)  # Display the processed frame

        # Show the map with overlay
        cv2.imshow("Map with Density", updated_map)

    # Wait for key press to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
>>>>>>> 57eb7f1fa34fa39deb7752ce475d84cc8b7dafe6
