"""
Object Detection using YOLOv8
Author: Saran Kumar S
Internship Project - CodeAlpha AI Internship
"""

import cv2
from ultralytics import YOLO
import argparse


def run_detection(source=0, model_path="yolov8n.pt", conf=0.5, save=False):
    """
    Run object detection on webcam or video file.

    Args:
        source     : 0 for webcam, or path to video file (e.g., 'video.mp4')
        model_path : YOLO model to use (yolov8n.pt is the lightest)
        conf       : Confidence threshold (0.0 - 1.0)
        save       : Save output video if True
    """

    # Load YOLO model
    model = YOLO(model_path)
    print(f"[INFO] Model loaded: {model_path}")
    print(f"[INFO] Starting detection on source: {source}")
    print("[INFO] Press 'q' to quit\n")

    # Open video/webcam
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source. Check your webcam or file path.")
        return

    # Video writer setup (only if saving)
    writer = None
    if save:
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        writer = cv2.VideoWriter(
            "output_detection.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            fps,
            (width, height)
        )
        print("[INFO] Saving output to output_detection.avi")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video ended or frame not received.")
            break

        # Run YOLO detection
        results = model(frame, conf=conf, verbose=False)

        # Draw bounding boxes on frame
        annotated_frame = results[0].plot()

        # Show object count on frame
        count = len(results[0].boxes)
        cv2.putText(
            annotated_frame,
            f"Objects Detected: {count}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display
        cv2.imshow("Object Detection - CodeAlpha Internship", annotated_frame)

        if writer:
            writer.write(annotated_frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit signal received.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument("--source", default=0,         help="0 for webcam or path to video file")
    parser.add_argument("--model",  default="yolov8n.pt", help="YOLO model file (default: yolov8n.pt)")
    parser.add_argument("--conf",   default=0.5, type=float, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--save",   action="store_true",  help="Save output video")
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    run_detection(source=source, model_path=args.model, conf=args.conf, save=args.save)
