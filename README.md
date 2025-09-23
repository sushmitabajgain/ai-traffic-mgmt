---

# AI Traffic management project

---

## Install dependency:
```
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Yolo/Ultralytics Setup:
```
cd src

python test_detect.py
```
We should see bounding boxes on cars/buses/trucks.

If that works, YOLO is set up.

---

```
cd src

python main.py
```

## What we achieved?

YOLO vehicle detection ✅

Lane-wise counting ✅

Adaptive green timing with live panel ✅


---

### It’s a real-time traffic-signal demo that:

- reads a video (or webcam),

- detects vehicles (YOLO), tracks them (IDs),

- assigns each to a lane polygon (and optional queue ROI near the stop line),

- turns those into demand numbers,

- runs two controllers in parallel:

    - Adaptive (uses demand, re-plans each cycle)

    - Fixed (round-robin baseline, ignores demand)

- logs metrics and renders a dashboard: left = data, right = video with boxes.

### What each file does

- ultralytics_detector.py – wraps Ultralytics YOLO (e.g., yolov8n.pt).
Returns detections as dicts with bbox, center, class, conf.

- tracker.py – a simple nearest-center tracker that gives each vehicle a stable ID and majority-voted class (stable_class) to reduce bus/truck flips.

- lane_counter.py – loads lanes.json with:

    a lane polygon (detection anywhere in it counts as occupancy), and

    an optional queue_polygon (near stop line; detections in it count as queue).

- traffic_logic.py – the AdaptiveSignalController:

    - smooths per-group demand with an EMA,

    - computes a green plan each cycle within [min_green, max_green],

    - switches phases when the active green ends,

    - exposes state() → {active_group, green_plan, phase_remaining, density_ema}.

- fixed_controller.py – the fixed-time baseline controller (round-robin split).

- metrics_logger.py – logs per-frame:

    - per-group occupancy and queue,

    - red-queue proxy (sum of queued vehicles in the non-active groups),

    - keeps running totals and prints a % improvement (adaptive vs fixed) at exit.

- main.py – wires it all together, renders the UI, and runs the loop.

## The main loop (what happens every frame)

```
capture → detect → track → assign → count → control → log → draw → show
```

### Capture
open_smart(...) opens your preferred MP4 (or falls back to webcam) using a backend that works on your OS.

### Detect
YOLO returns boxes/classes for vehicles in the frame.

### Track
SimpleTracker.update() matches detections to existing tracks (IDs), updates their positions, and votes on class over time → stable_class.

### Assign to lanes
For each track center (cx, cy), LaneManager.assign() finds the containing lane polygon, and in_queue() checks the queue polygon.

### Count (with weights)
Each track is bucketed to two_w / light / heavy (bicycle/moto, car/van, bus/truck) and given a weight (0.5 / 1.0 / 2.0).

Occupancy per lane += weight if in the lane polygon

Queue per lane += weight if also in the queue ROI

### Build demand
- For each lane:
    demand = occupancy * OCC_ALPHA + queue * QUEUE_BETA (defaults: 1.0 and 2.0).
    Demand is aggregated per group (NS / EW).

### Control

Adaptive: uses smoothed demand to set green times for NS/EW for the next cycle, constrained by min_green/max_green. It tracks active group and seconds remaining this phase.

Fixed: just rotates on a fixed split (baseline for comparison).

### Log metrics
For both controllers, compute red-queue this frame = queued vehicles in the other (non-active) group. Accumulate over time and write to data/metrics.csv.
At shutdown, print totals and Improvement %.

### Draw dashboard
- Right: frame with tracked boxes + labels #id stable_class/bucket.
- Left (dark panel):

    Adaptive Active / Fixed Active chips

    Planned green bars for NS/EW (adaptive plan for next cycle)

    Demand by group bars (occupancy & queue sums)

    Red-queue sparklines (adaptive vs fixed over time)

    Compact per-lane list: occ / q [group]
    Then it stitches panel | video and shows in one window.

### Exit
Cleanly closes camera/video, window, and prints the metrics summary.

## Why the design choices matter

Tracker + class voting → stable IDs/labels; avoids bus↔truck flips from impacting control.

Bucket weights → larger vehicles contribute more demand, without needing perfect class.

Queue ROI → prioritizes stopped vehicles at the stop line, not just mid-block traffic.

Two controllers in parallel → you can prove the adaptive approach beats a fixed plan on the same video.
