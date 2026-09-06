import cv2
import numpy as np
import requests
from ultralytics import YOLO

ESP_IP = "192.168.0.172"  # Change if your ESP IP changes
STREAM_URL = f"http://{ESP_IP}/stream"

# yolo11n = smallest, suitable for CPU testing
model = YOLO("yolo11n.pt")

session = requests.Session()
session.trust_env = False
response = None

try:
    print("Connecting to ESP32-CAM...")

    response = session.get(
        STREAM_URL,
        stream=True,
        timeout=(5, 60)
    )

    response.raise_for_status()

    buffer = b""

    for chunk in response.iter_content(chunk_size=4096):
        if not chunk:
            continue

        buffer += chunk

        while True:
            jpg_start = buffer.find(b"\xff\xd8")
            jpg_end = buffer.find(b"\xff\xd9", jpg_start + 2)

            if jpg_start == -1 or jpg_end == -1:
                break

            jpg = buffer[jpg_start:jpg_end + 2]
            buffer = buffer[jpg_end + 2:]

            frame = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                continue

            # PC runs AI on the current ESP32 image
            results = model(
                frame,
                imgsz=416,
                conf=0.25,
                verbose=False,
                device="cpu"
            )

            # Add YOLO boxes, labels, and confidence values
            detected_frame = results[0].plot()

            cv2.imshow(
                "ESP32-CAM + YOLO Object Detection",
                detected_frame
            )

            # Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print("Stopped.")

except Exception as error:
    print("Error:", error)

finally:
    if response is not None:
        response.close()

    cv2.destroyAllWindows()
