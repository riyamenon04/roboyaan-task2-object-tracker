# Roboyaan Challenge — Task 2
## Real-Time Target Object Detection and Tracking

---

## Overview
A real-time object detection and tracking system
where the user types any object name, and the system
instantly finds and tracks it in the live camera feed
using a bounding box — ignoring everything else.

---

## How It Works
1. User types the target object name
2. YOLO scans every camera frame
3. Only matching objects are kept
4. SORT tracker assigns ID and follows the object
5. Bounding box drawn around the target only


---

## Output Format
```
Target Object : Bottle
Confidence    : 96%
Status        : Tracking Active
FPS           : 10.1
```

---

## Tracking Behavior

| Condition | Action |
|---|---|
| Object matches target | Track + show bounding box |
| Object does not match | Ignore completely |
| Target lost | Show Searching + redetect |
| Target reappears | Resume tracking automatically |

---

## Tools Used

| Tool | Purpose |
|---|---|
| Python | Primary language |
| YOLOv8s | Object detection |
| SORT | Object tracking across frames |
| OpenCV | Camera feed + bounding box display |

---

## How to Run

### Requirements
```bash
pip install ultralytics opencv-python numpy
pip install filterpy scikit-image lap
git clone https://github.com/abewley/sort.git
cp sort/sort.py .
```

### Run
```bash
python3 tracker.py
```

### Supported Objects
Any of the 80 COCO classes including:
```
person, bottle, book, laptop, cell phone,
cup, chair, keyboard, mouse, backpack
```

---
