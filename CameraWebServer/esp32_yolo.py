import cv2
import requests
import numpy as np
from ultralytics import YOLO

URL = "http://192.168.0.172:81/stream"
model = YOLO("yolo11n.pt")  # lightweight model; downloads once if needed

session = requests.Session()
session.trust_env = False
response = None

try:
    print("Connecting to ESP32-CAM...")
    response = session.get(URL, stream=True, timeout=(5, 60))
    response.raise_for_status()

    buffer = b""

    for chunk in response.iter_content(chunk_size=4096):
        if not chunk:
            continue

        buffer += chunk

        while True:
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start + 2)

            if start == -1 or end == -1:
                break

            jpg = buffer[start:end + 2]
            buffer = buffer[end + 2:]

            frame = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                continue

            # Run object detection on this camera frame
            results = model(frame, imgsz=416, conf=0.45, verbose=False)

            # Draw boxes, labels, and confidence values
            detected_frame = results[0].plot()

            cv2.imshow("ESP32-CAM Object Detection", detected_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print("Stopped.")

except Exception as e:
    print("Error:", e)

finally:
    if response is not None:
        response.close()
    cv2.destroyAllWindows()