"""Local browser dashboard for OpenCV object detection and distance telemetry."""

import argparse
import importlib.util
import threading
import time
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "edge_&_object_detection - Copy.py"
spec = importlib.util.spec_from_file_location("vision_source", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load vision source: {SOURCE}")
vision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vision)

app = Flask(__name__)
state_lock = threading.Lock()
latest_jpeg = None
latest_data = {
    "status": "starting",
    "timestamp": 0,
    "fps": 0,
    "inference_ms": 0,
    "inference_age": 0,
    "objects": [],
    "distances": [],
    "error": None,
}

REAL_WIDTHS = {
    "person": 0.45,
    "cell phone": 0.075,
    "car": 1.8,
    "truck": 2.5,
    "bus": 2.5,
    "bicycle": 0.6,
    "motorcycle": 0.8,
}


def set_state(**values):
    with state_lock:
        latest_data.update(values)


def estimate_object_distance(width_pixels, class_name, focal_length):
    real_width = REAL_WIDTHS.get(class_name)
    if real_width is None or width_pixels <= 0:
        return None
    return vision.estimate_distance(width_pixels, real_width, focal_length)


def draw_objects(frame, objects, distances):
    annotated = frame.copy()
    object_by_id = {item["id"]: item for item in objects}

    for pair in distances:
        first = object_by_id.get(pair["from_id"])
        second = object_by_id.get(pair["to_id"])
        if first is None or second is None:
            continue
        first_center = tuple(map(int, first["center"]))
        second_center = tuple(map(int, second["center"]))
        midpoint = (
            (first_center[0] + second_center[0]) // 2,
            (first_center[1] + second_center[1]) // 2,
        )
        cv2.line(annotated, first_center, second_center, (255, 190, 55), 3)
        cv2.circle(annotated, first_center, 6, (255, 255, 255), -1)
        cv2.circle(annotated, second_center, 6, (255, 255, 255), -1)
        distance_label = f'{pair["meters"]:.2f} m'
        (label_width, label_height), _ = cv2.getTextSize(
            distance_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            annotated,
            (midpoint[0] - 8, midpoint[1] - label_height - 12),
            (midpoint[0] + label_width + 8, midpoint[1] + 6),
            (5, 10, 12),
            -1,
        )
        cv2.putText(
            annotated,
            distance_label,
            (midpoint[0], midpoint[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 220, 100),
            2,
            cv2.LINE_AA,
        )

    for item in objects:
        x1, y1, x2, y2 = item["box"]
        label = f'{item["label"]} {item["confidence"]:.0%}'
        depth = item["distance_from_camera_m"]
        if depth is not None:
            label += f" {depth:.2f}m"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 120), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 120),
            2,
            cv2.LINE_AA,
        )
    return annotated


def make_detection(detector, frame, confidence, focal_length, inference_size):
    result = detector.predict(
        frame, conf=confidence, imgsz=inference_size, device="cpu", verbose=False
    )[0]
    objects = []
    names = result.names

    for box, class_id, score in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
        result.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = map(int, box)
        class_name = names[class_id]
        width = max(x2 - x1, 1)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        depth = estimate_object_distance(width, class_name, focal_length)
        objects.append(
            {
                "id": len(objects),
                "label": class_name,
                "confidence": round(float(score), 3),
                "box": [x1, y1, x2, y2],
                "center": [round(center_x, 1), round(center_y, 1)],
                "distance_from_camera_m": (
                    round(depth, 3) if depth is not None else None
                ),
                "known_size": class_name in REAL_WIDTHS,
            }
        )

    distances = []
    for first_index, first in enumerate(objects):
        if first["distance_from_camera_m"] is None:
            continue
        for second in objects[first_index + 1 :]:
            if second["distance_from_camera_m"] is None:
                continue
            first_depth = first["distance_from_camera_m"]
            second_depth = second["distance_from_camera_m"]
            pixel_dx = first["center"][0] - second["center"][0]
            pixel_dy = first["center"][1] - second["center"][1]
            average_depth = (first_depth + second_depth) / 2
            lateral = pixel_dx * average_depth / focal_length
            vertical = pixel_dy * average_depth / focal_length
            depth_difference = first_depth - second_depth
            separation = (lateral**2 + vertical**2 + depth_difference**2) ** 0.5
            distances.append(
                {
                    "from_id": first["id"],
                    "to_id": second["id"],
                    "from": first["label"],
                    "to": second["label"],
                    "meters": round(separation, 3),
                }
            )

    return draw_objects(frame, objects, distances), objects, distances


def camera_worker(
    camera_index,
    model_path,
    confidence,
    focal_length,
    width,
    height,
    infer_every,
    inference_size,
):
    global latest_jpeg
    camera = None
    try:
        camera = vision.open_camera(camera_index, width, height)
        set_state(status="loading model", error=None)
        detector = vision.load_detector(model_path)
        set_state(status="live")

        frame_number = 0
        last_objects = []
        last_distances = []
        last_inference_time = time.perf_counter()
        frames_since_inference = 0
        fps_started = time.perf_counter()
        fps_frames = 0

        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera.")
            frame_number += 1
            frames_since_inference += 1
            fps_frames += 1
            inference_ms = latest_data["inference_ms"]
            if frame_number == 1 or frame_number % infer_every == 0:
                started = time.perf_counter()
                annotated, last_objects, last_distances = make_detection(
                    detector,
                    frame,
                    confidence,
                    focal_length,
                    inference_size,
                )
                inference_ms = (time.perf_counter() - started) * 1000
                frames_since_inference = 0
            else:
                annotated = draw_objects(frame, last_objects, last_distances)

            elapsed = time.perf_counter() - fps_started
            current_fps = fps_frames / elapsed if elapsed else 0
            ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                continue
            with state_lock:
                latest_jpeg = encoded.tobytes()
                latest_data.update(
                    {
                        "timestamp": time.time(),
                        "objects": last_objects,
                        "distances": last_distances,
                        "fps": round(current_fps, 1),
                        "inference_ms": round(inference_ms, 1),
                        "inference_age": frames_since_inference,
                        "error": None,
                    }
                )
    except Exception as error:
        set_state(status="error", error=str(error))
    finally:
        if camera is not None:
            camera.release()


def video_stream():
    while True:
        with state_lock:
            frame = latest_jpeg
        if frame is not None:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
        time.sleep(0.03)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video")
def video():
    return Response(video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/state")
def api_state():
    with state_lock:
        return jsonify(latest_data)


def main():
    parser = argparse.ArgumentParser(description="Run the local OpenCV vehicle dashboard.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default=str(ROOT / "yolo11n.pt"))
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--focal-length", type=float, default=700.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--infer-every", type=int, default=4)
    parser.add_argument("--inference-size", type=int, default=416)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.inference_size < 160 or args.inference_size > 1280:
        parser.error("--inference-size must be between 160 and 1280")
    if args.infer_every < 1:
        parser.error("--infer-every must be at least 1")

    worker = threading.Thread(
        target=camera_worker,
        args=(
            args.camera,
            args.model,
            args.confidence,
            args.focal_length,
            args.width,
            args.height,
            max(1, args.infer_every),
            args.inference_size,
        ),
        daemon=True,
    )
    worker.start()
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
