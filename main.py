import cv2
import torch
import json

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


def convert_to_yolo_format(result, width, height):
    yolo_results = []
    for detection in result.pandas().xyxy[0].to_dict(orient="records"):
        label = detection['name']
        confidence = float(detection['confidence'])
        # convert bbox from pixels to YOLO format
        x_center = (detection['xmin'] + detection['xmax']) / 2
        y_center = (detection['ymin'] + detection['ymax']) / 2
        bbox_width = detection['xmax'] - detection['xmin']
        bbox_height = detection['ymax'] - detection['ymin']
        bbox = [x_center / width, y_center / height,
                bbox_width / width, bbox_height / height]

        yolo_results.append({"label": label,"confidence": confidence,"bbox": bbox})

    return yolo_results


def save_yolo_results_to_json(yolo_results, output_file):
    with open(output_file, 'a') as f:
        json.dump(yolo_results, f)


def stream_rtsp(rtsp_url, output_file):
    cap = cv2.VideoCapture(rtsp_url)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to receive frame.")
            break

        results = model(frame)

        cv2.imshow('RTSP Stream', results.render()[0])

        # Convert YOLOv5 results to YOLO-like format
        height, width = frame.shape[:2]
        yolo_results = convert_to_yolo_format(results, width, height)

        # Save YOLO-like results to JSON
        save_yolo_results_to_json(yolo_results, output_file)

        # Press 'q' to exit the loop and stop streaming
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return results


if __name__ == "__main__":
    rtsp_url = 'https://5e0da72d486c5.streamlock.net:8443/ayalon/HaShalom.stream/playlist.m3u8'  # Replace with your RTSP URL
    output_file = 'yolo_results.json'  # Output JSON file for YOLO-like results
    res = stream_rtsp(rtsp_url, output_file)
