# ============================================
#   ROBOYAAN CHALLENGE — TASK 2
#   Target Object Detection and Tracking
# ============================================

import cv2
from ultralytics import YOLO
from sort import Sort
import numpy as np
import time

# ── Load YOLO model ──
model = YOLO("yolov8s.pt")

# ── Start SORT tracker ──
tracker = Sort(max_age=60, min_hits=1, iou_threshold=0.2)


# ── Ask user input BEFORE camera starts ──
target = input("Enter target object to track: ").strip().lower()
print(f"\n Target set: '{target}'")
print("Press Q on keyboard after clicking the camera window to quit")
print("=" * 45)
print(f"   ROBOYAAN TASK 2 — OBJECT TRACKER")
print("=" * 45)

# ── Start camera ──
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Camera not found!")
    exit()

# ── Variables ──
prev_time  = time.time()
last_status = None

# ── Main loop ──
while True:
    ret, frame = cap.read()
    if not ret:
        print(" Cannot read frame")
        break

    # ── FPS ──
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # ── YOLO detects ALL objects ──
    results = model(frame, verbose=False, conf=0.15)

    # ── Filter — keep only target ──
    detections_for_sort = []
    current_confidence  = 0

    for result in results:
        for box in result.boxes:
            class_id   = int(box.cls[0])
            class_name = model.names[class_id].lower()
            confidence = float(box.conf[0])

            if class_name == target:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections_for_sort.append(
                    [x1, y1, x2, y2, confidence]
                )
                current_confidence = confidence

    # ── SORT tracking ──
    detections_array = np.array(detections_for_sort) if detections_for_sort \
                       else np.empty((0, 5))
    tracked_objects  = tracker.update(detections_array)

    # ── Status ──
    if len(tracked_objects) > 0:
        status = "Tracking Active"
    else:
        status = "Searching for target..."

    # ── Console — print only when status changes ──
    if status != last_status:
        print(f"\nTarget Object  : {target.capitalize()}")
        if current_confidence > 0:
            print(f"Confidence     : {current_confidence:.0%}")
        print(f"Status         : {status}")
        print(f"FPS            : {fps:.1f}")
        print("-" * 45)
        last_status = status

    # ── Draw bounding boxes ──
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)

        # Green rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Label on box
        label = f"{target.capitalize()} | ID:{int(track_id)} | {current_confidence:.0%}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ── On screen info ──
    # FPS — top left yellow
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Target — below FPS
    cv2.putText(frame, f"Target: {target.capitalize()}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Status — green if tracking, red if searching
    status_color = (0, 255, 0) if status == "Tracking Active" else (0, 0, 255)
    cv2.putText(frame, status, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # Quit reminder — bottom of screen
    cv2.putText(frame, "Click window + Press Q to quit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── Show frame ──
    cv2.imshow("Roboyaan Task 2 — Object Tracker", frame)

    # ── Q to quit ──
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ──
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 45)
print("   Tracker stopped successfully")
print("=" * 45)