"""
Object Tracking using YOLOv8 + ByteTrack
Author: Saran Kumar S
Internship Project - CodeAlpha AI Internship
"""

import cv2
from ultralytics import YOLO
import argparse
from collections import defaultdict


def run_tracking(source=0, model_path="yolov8n.pt", conf=0.4, save=False):
    """
    Run object tracking with unique IDs on webcam or video file.

    Args:
        source     : 0 for webcam, or path to video file (e.g., 'video.mp4')
        model_path : YOLO model to use
        conf       : Confidence threshold (0.0 - 1.0)
        save       : Save output video if True
    """

    model = YOLO(model_path)
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Starting tracking on source: {source}")
    print("[INFO] Press 'q' to quit\n")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source.")
        return

    # Store track history for trail lines
    track_history = defaultdict(list)

    writer = None
    if save:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        writer = cv2.VideoWriter(
            "output_tracking.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (width, height)
        )
        print("[INFO] Saving output to output_tracking.avi")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video ended or frame not received.")
            break

        # Run YOLOv8 tracking (ByteTrack)
        results = model.track(frame, conf=conf, persist=True, tracker="bytetrack.yaml", verbose=False)

        annotated_frame = results[0].plot()

        # Draw movement trails for each tracked object
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                cx, cy = int(x), int(y)

                # Store center point history
                track_history[track_id].append((cx, cy))
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                # Draw trail line
                points = track_history[track_id]
                for i in range(1, len(points)):
                    cv2.line(annotated_frame, points[i - 1], points[i], (230, 230, 0), 2)

        # Show stats
        active_ids = len(results[0].boxes.id) if results[0].boxes.id is not None else 0
        cv2.putText(
            annotated_frame,
            f"Tracking {active_ids} object(s)",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 200, 255),
            2
        )

        cv2.imshow("Object Tracking - CodeAlpha Internship", annotated_frame)

        if writer:
            writer.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit signal received.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Tracking complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Object Tracking")
    parser.add_argument("--source", default=0,            help="0 for webcam or path to video file")
    parser.add_argument("--model",  default="yolov8n.pt", help="YOLO model file")
    parser.add_argument("--conf",   default=0.4, type=float, help="Confidence threshold")
    parser.add_argument("--save",   action="store_true",   help="Save output video")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    run_tracking(source=source, model_path=args.model, conf=args.conf, save=args.save)
