import time
import threading

import cv2
import numpy as np
import requests
from ultralytics import YOLO


# ============================================================
# ESP32-CAM URLs
# ============================================================

ESP_IP = "192.168.0.172"

# Camera stream is now on port 81.
STREAM_URL = f"http://{ESP_IP}:81/stream"

# GPIO commands remain on normal HTTP port 80.
PERSON_ON_URL = f"http://{ESP_IP}/person_on"
PERSON_OFF_URL = f"http://{ESP_IP}/person_off"


# ============================================================
# YOLO settings for CPU
# ============================================================

MODEL_NAME = "yolo11n.pt"
YOLO_IMAGE_SIZE = 320
YOLO_CONFIDENCE = 0.25

# COCO class ID 0 means person.
PERSON_CLASS_ID = 0


# ============================================================
# Stable detection settings
# ============================================================

# Person must appear in two frames before LED turns ON.
PERSON_ON_FRAMES = 2

# Person must disappear for eight frames before LED turns OFF.
PERSON_OFF_FRAMES = 8


# ============================================================
# Program state
# ============================================================

last_person_state = None

person_seen_frames = 0
person_missing_frames = 0
stable_person_state = False

latest_frame = None
latest_frame_id = 0
frame_lock = threading.Lock()

stop_event = threading.Event()

control_session = requests.Session()
control_session.trust_env = False


# ============================================================
# Send command to ESP32 only when state changes
# ============================================================

def send_person_command(person_present, force=False):
    global last_person_state

    # Do not send repeated HTTP requests every frame.
    if not force and person_present == last_person_state:
        return

    url = PERSON_ON_URL if person_present else PERSON_OFF_URL

    try:
        response = control_session.get(url, timeout=2)
        response.raise_for_status()

        # Store state only after ESP32 responds successfully.
        last_person_state = person_present

        if person_present:
            print("ESP32: PERSON ON -> GPIO13 HIGH")
        else:
            print("ESP32: PERSON OFF -> GPIO12/13 LOW")

    except requests.RequestException as error:
        print("ESP32 HTTP command failed:", error)


# ============================================================
# Prevent LED flickering caused by missed YOLO frames
# ============================================================

def update_stable_person_state(person_detected_this_frame):
    global person_seen_frames
    global person_missing_frames
    global stable_person_state

    if person_detected_this_frame:
        person_seen_frames += 1
        person_missing_frames = 0

        if person_seen_frames >= PERSON_ON_FRAMES:
            stable_person_state = True

    else:
        person_missing_frames += 1
        person_seen_frames = 0

        if person_missing_frames >= PERSON_OFF_FRAMES:
            stable_person_state = False

    return stable_person_state


# ============================================================
# Receive camera video in a background thread
# ============================================================

def camera_reader():
    global latest_frame
    global latest_frame_id

    session = requests.Session()
    session.trust_env = False

    while not stop_event.is_set():
        response = None
        jpeg_buffer = b""

        try:
            print("Connecting to ESP32 camera stream...")

            response = session.get(
                STREAM_URL,
                stream=True,
                timeout=(5, 60)
            )

            response.raise_for_status()

            print("ESP32 camera stream connected.")

            for chunk in response.iter_content(chunk_size=4096):
                if stop_event.is_set():
                    break

                if not chunk:
                    continue

                jpeg_buffer += chunk

                while True:
                    jpg_start = jpeg_buffer.find(b"\xff\xd8")

                    if jpg_start == -1:
                        break

                    jpg_end = jpeg_buffer.find(b"\xff\xd9", jpg_start + 2)

                    if jpg_end == -1:
                        break

                    jpg = jpeg_buffer[jpg_start:jpg_end + 2]
                    jpeg_buffer = jpeg_buffer[jpg_end + 2:]

                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )

                    if frame is None:
                        continue

                    # Keep only newest frame, preventing lag.
                    with frame_lock:
                        latest_frame = frame
                        latest_frame_id += 1

        except requests.RequestException as error:
            print("ESP32 stream error:", error)
            time.sleep(2)

        finally:
            if response is not None:
                response.close()

    session.close()


# ============================================================
# Main application
# ============================================================

try:
    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    # Ensure LED starts OFF.
    send_person_command(False, force=True)

    reader_thread = threading.Thread(
        target=camera_reader,
        daemon=True
    )

    reader_thread.start()

    print("Press Q in the video window to stop.")

    last_processed_frame_id = -1
    last_raw_person_state = None

    fps_start_time = time.time()
    fps_frames = 0
    displayed_fps = 0.0

    while True:
        with frame_lock:
            current_frame = (
                latest_frame.copy()
                if latest_frame is not None
                else None
            )

            current_frame_id = latest_frame_id

        if current_frame is None or current_frame_id == last_processed_frame_id:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            time.sleep(0.005)
            continue

        last_processed_frame_id = current_frame_id

        # Detect only the COCO "person" class.
        results = model(
            current_frame,
            imgsz=YOLO_IMAGE_SIZE,
            conf=YOLO_CONFIDENCE,
            classes=[PERSON_CLASS_ID],
            verbose=False,
            device="cpu"
        )

        boxes = results[0].boxes

        person_detected_this_frame = (
            boxes is not None and len(boxes) > 0
        )

        # Print only when raw YOLO result changes.
        if person_detected_this_frame != last_raw_person_state:
            print(
                "YOLO person in frame:",
                person_detected_this_frame
            )

            last_raw_person_state = person_detected_this_frame

        current_person_state = update_stable_person_state(
            person_detected_this_frame
        )

        # Sends /person_on or /person_off only on state change.
        send_person_command(current_person_state)

        output_frame = results[0].plot()

        fps_frames += 1
        elapsed_time = time.time() - fps_start_time

        if elapsed_time >= 1.0:
            displayed_fps = fps_frames / elapsed_time
            fps_frames = 0
            fps_start_time = time.time()

        if current_person_state:
            status_text = "PERSON CONFIRMED - GPIO13 HIGH"
            status_colour = (0, 0, 255)
        else:
            status_text = "NO PERSON - GPIO12/13 LOW"
            status_colour = (0, 255, 0)

        cv2.putText(
            output_frame,
            status_text,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_colour,
            2
        )

        cv2.putText(
            output_frame,
            f"FPS: {displayed_fps:.1f}",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.imshow(
            "ESP32-CAM YOLO Person Detection",
            output_frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nStopped by user.")

except Exception as error:
    print("\nUnexpected error:", error)

finally:
    stop_event.set()

    # Turn LED OFF when program stops.
    send_person_command(False, force=True)

    cv2.destroyAllWindows()
    control_session.close()

    print("Program closed safely.")