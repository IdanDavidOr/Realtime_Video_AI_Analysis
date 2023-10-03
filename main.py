import cv2
import torch
import json
import threading
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Global variables to store inference results and metadata
inference_results = None
metadata = None

def convert_to_yolo_format(result, width, height):
    yolo_results = []
    for detection in result.pandas().xyxy[0].to_dict(orient="records"):
        label = detection['name']
        confidence = float(detection['confidence'])
        bbox = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']

        # Convert bbox to YOLO relative format
        x_center = (bbox[0] + bbox[2]) / (2 * width)
        y_center = (bbox[1] + bbox[3]) / (2 * height)
        bbox_width = (bbox[2] - bbox[0]) / width
        bbox_height = (bbox[3] - bbox[1]) / height

        yolo_results.append({
            "label": label,
            "confidence": confidence,
            "bbox": [x_center, y_center, bbox_width, bbox_height]
        })
    return yolo_results

def save_to_json(output_file):
    global inference_results, metadata

    while True:
        if inference_results is not None:
            # Combine inference results and metadata
            combined_data = {"inference_results": inference_results, "metadata": metadata}

            # Save to JSON
            with open(output_file, 'a') as f:
                json.dump(combined_data, f, indent=4)

            # Reset inference results and metadata
            inference_results = None
            metadata = None

        time.sleep(1)  # Adjust sleep time as needed

def stream_rtsp(rtsp_url, output_file):
    global inference_results, metadata
    cap = cv2.VideoCapture(rtsp_url)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to receive frame.")
            break

        results = model(frame)

        # Convert YOLOv5 results to YOLO relative format
        height, width = frame.shape[:2]
        yolo_results = convert_to_yolo_format(results, width, height)

        # Store inference results and metadata
        inference_results = yolo_results
        metadata = {"timestamp": time.time()}  # Example metadata, you can modify this based on your requirements

        cv2.imshow('RTSP Stream', results.render()[0])

        # Press 'q' to exit the loop and stop streaming
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return results

if __name__ == "__main__":
    rtsp_url = 'https://5e0da72d486c5.streamlock.net:8443/ayalon/HaShalom.stream/playlist.m3u8'  # Replace with your RTSP URL
    output_file = 'yolo_results.json'  # Output JSON file for YOLO relative format results

    # Create a thread for writing to JSON
    json_thread = threading.Thread(target=save_to_json, args=(output_file,))
    json_thread.daemon = True
    json_thread.start()

    # Run the main thread for streaming and inference
    stream_rtsp(rtsp_url, output_file)
