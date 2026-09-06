"""Live stream object detection and edge detection with OpenCV.

Examples:
	python edge_detection.py
	python edge_detection.py --left 0 --right 1
	python edge_detection.py --side-by-side 0
"""

import argparse
import sys

import cv2


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Display live YOLO object detection beside Canny edge detection."
	)
	parser.add_argument(
		"--left",
		type=int,
		default=0,
		help="Camera index for the left camera (default: 0).",
	)
	parser.add_argument(
		"--right",
		type=int,
		default=None,
		help="Camera index for the right camera; set this for two-camera stereo.",
	)
	parser.add_argument(
		"--side-by-side",
		type=int,
		metavar="CAMERA",
		help="Use one camera that outputs left and right images side by side.",
	)
	parser.add_argument(
		"--single",
		type=int,
		metavar="CAMERA",
		help="Use one normal camera and display its live edge detection.",
	)
	parser.add_argument(
		"--width",
		type=int,
		default=1280,
		help="Requested capture width per camera (default: 1280).",
	)
	parser.add_argument(
		"--height",
		type=int,
		default=480,
		help="Requested capture height (default: 480).",
	)
	parser.add_argument(
		"--low-threshold",
		type=int,
		default=50,
		help="Lower Canny threshold (default: 50).",
	)
	parser.add_argument(
		"--high-threshold",
		type=int,
		default=150,
		help="Upper Canny threshold (default: 150).",
	)
	parser.add_argument(
		"--model",
		default="yolo11n.pt",
		help="Ultralytics YOLO model (default: yolo11n.pt).",
	)
	parser.add_argument(
		"--confidence",
		type=float,
		default=0.35,
		help="Minimum object confidence from 0 to 1 (default: 0.35).",
	)
	parser.add_argument(
		"--focal-length",
		type=float,
		default=700.0,
		help="Camera focal length in pixels for distance estimation (default: 700).",
	)
	return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
	backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
	camera = cv2.VideoCapture(index, backend)
	camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
	if not camera.isOpened():
		camera.release()
		raise RuntimeError(f"Could not open camera {index}.")
	return camera


def detect_edges(
	frame: cv2.Mat, low_threshold: int, high_threshold: int
) -> cv2.Mat:
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	return cv2.Canny(blurred, low_threshold, high_threshold)


def add_label(frame: cv2.Mat, text: str) -> cv2.Mat:
	labeled = frame.copy()
	cv2.putText(
		labeled,
		text,
		(15, 30),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.8,
		(0, 255, 0),
		2,
		cv2.LINE_AA,
	)
	return labeled


def load_detector(model_name: str):
	try:
		from ultralytics import YOLO
	except ImportError as error:
		raise RuntimeError(
			"Object detection requires ultralytics. Install it with: "
			"pip install ultralytics"
		) from error
	return YOLO(model_name)


def estimate_distance(object_width_pixels: float, real_width_meters: float, focal_length: float) -> float:
	if object_width_pixels <= 0:
		return 0.0
	return (real_width_meters * focal_length) / object_width_pixels


def detect_objects(
	detector, frame: cv2.Mat, confidence: float, focal_length: float
) -> cv2.Mat:
	results = detector.predict(frame, conf=confidence, imgsz=640, verbose=False)
	result = results[0]
	annotated = frame.copy()
	class_names = result.names

	for box, class_id, score in zip(
		result.boxes.xyxy.cpu().numpy(),
		result.boxes.cls.cpu().numpy().astype(int),
		result.boxes.conf.cpu().numpy(),
	):
		x1, y1, x2, y2 = map(int, box)
		class_name = class_names[class_id]
		if class_name == "person":
			real_width = 0.45
		elif class_name == "cell phone":
			real_width = 0.075
		else:
			real_width = None

		if real_width is None:
			label = f"{class_name} {score:.0%}"
		else:
			distance = estimate_distance(
				max(x2 - x1, 1), real_width, focal_length
			)
			label = f"{class_name} {score:.0%} {distance:.2f} m"

		cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
		text_y = max(y1 - 10, 25)
		cv2.putText(
			annotated,
			label,
			(x1, text_y),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.55,
			(0, 255, 0),
			2,
			cv2.LINE_AA,
		)

	return annotated


def main() -> int:
	args = parse_args()
	cameras = []
	single_mode = args.single is not None

	try:
		if args.low_threshold < 0 or args.high_threshold <= args.low_threshold:
			raise ValueError("high threshold must be greater than low threshold")
		if not 0 < args.confidence <= 1:
			raise ValueError("confidence must be greater than 0 and at most 1")

		if args.side_by_side is not None and args.single is not None:
			raise ValueError("choose either --side-by-side or --single, not both")

		if args.side_by_side is not None:
			cameras.append(open_camera(args.side_by_side, args.width * 2, args.height))
		elif args.single is not None:
			cameras.append(open_camera(args.single, args.width, args.height))
		else:
			cameras.append(open_camera(args.left, args.width, args.height))
			if args.right is not None:
				cameras.append(open_camera(args.right, args.width, args.height))
			else:
				single_mode = True

		cv2.namedWindow("Object and edge detection", cv2.WINDOW_NORMAL)
		ok, preview = cameras[0].read()
		if ok:
			preview = add_label(preview, "Loading object detector...")
			cv2.imshow("Object and edge detection", preview)
			cv2.waitKey(1)
		print(f"Loading YOLO model: {args.model}", flush=True)
		detector = load_detector(args.model)
		print("YOLO model ready. Starting live detection.", flush=True)

		while True:
			ok, first_frame = cameras[0].read()
			if not ok:
				print("Could not read from the stereo camera stream.", file=sys.stderr)
				break

			if single_mode:
				left_view = add_label(
					detect_objects(
						detector, first_frame, args.confidence, args.focal_length
					),
					"OBJECTS",
				)
				edges = detect_edges(
					first_frame, args.low_threshold, args.high_threshold
				)
				right_view = add_label(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), "EDGES")
			elif args.side_by_side is not None:
				midpoint = first_frame.shape[1] // 2
				left_frame = first_frame[:, :midpoint]
				right_frame = first_frame[:, midpoint:]
			else:
				ok, right_frame = cameras[1].read()
				if not ok:
					print("Could not read from the right camera.", file=sys.stderr)
					break
				left_frame = first_frame

			if not single_mode:
				left_edges = detect_edges(
					left_frame, args.low_threshold, args.high_threshold
				)
				right_edges = detect_edges(
					right_frame, args.low_threshold, args.high_threshold
				)
				left_objects = add_label(
					detect_objects(
						detector, left_frame, args.confidence, args.focal_length
					),
					"LEFT objects",
				)
				right_objects = add_label(
					detect_objects(
						detector, right_frame, args.confidence, args.focal_length
					),
					"RIGHT objects",
				)
				left_view = cv2.hconcat(
					[left_objects, add_label(cv2.cvtColor(left_edges, cv2.COLOR_GRAY2BGR), "LEFT edges")]
				)
				right_view = cv2.hconcat(
					[right_objects, add_label(cv2.cvtColor(right_edges, cv2.COLOR_GRAY2BGR), "RIGHT edges")]
				)

			stereo_view = cv2.hconcat([left_view, right_view])
			cv2.imshow("Object and edge detection", stereo_view)

			key = cv2.waitKey(1) & 0xFF
			if key in (ord("q"), 27):
				break
	except (RuntimeError, ValueError) as error:
		print(f"Error: {error}", file=sys.stderr)
		return 1
	finally:
		for camera in cameras:
			camera.release()
		cv2.destroyAllWindows()

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
