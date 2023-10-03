import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def stream_rtsp(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to receive frame.")
            break

        results = model(frame)

        cv2.imshow('RTSP Stream', results.render()[0])
        # cv2.imshow('RTSP Stream', frame)

        # Press 'q' to exit the loop and stop streaming
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return results



if __name__ == "__main__":
    rtsp_url = 'https://5e0da72d486c5.streamlock.net:8443/ayalon/HaShalom.stream/playlist.m3u8'  # Replace with your RTSP URL
    res = stream_rtsp(rtsp_url)

