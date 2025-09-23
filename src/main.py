# src/main.py
import os
import sys
from pathlib import Path
from collections import deque

import cv2
import numpy as np

from ultralytics_detector import UltralyticsPTDetector
from lane_counter import LaneManager
from traffic_logic import AdaptiveSignalController
from fixed_controller import FixedSignalController
from tracker import SimpleTracker
from metrics_logger import MetricsLogger

# ===================== Config =====================
# Prefer this file first; if not found we'll auto-pick the first *.mp4 in ../data
PREFERRED_VIDEO = str((Path(__file__).resolve().parents[1] / "data" / "sample.mp4").resolve())
LANES_JSON = str((Path(__file__).resolve().parents[1] / "lanes" / "lanes.json").resolve())

# Demand weighting (controller input): occupancy + k * queue
OCC_ALPHA = 1.0
QUEUE_BETA = 2.0

# Dashboard sizing
PANEL_W = 420
HIST_N = 150

# ===================== Smart video opener =====================
def _backend_names():
    names = {}
    for k in dir(cv2):
        if k.startswith("CAP_"):
            names[getattr(cv2, k)] = k
    return names

def _name_backend(be):
    return _backend_names().get(be, f"backend_{be}" if be else "AUTO")

def _platform_backends():
    if sys.platform.startswith("win"):
        return [cv2.CAP_FFMPEG, cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif sys.platform == "darwin":
        return [cv2.CAP_FFMPEG, cv2.CAP_AVFOUNDATION]
    else:
        return [cv2.CAP_FFMPEG, cv2.CAP_V4L2]

def _find_first_video_in_data():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    if data_dir.exists():
        for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            vids = list(data_dir.glob(ext))
            if vids:
                return str(vids[0].resolve())
    return None

def open_smart(prefer_path=None):
    """Try to open a file (preferred), else first video in ../data, else webcams 0..2."""
    tried = []
    backends = _platform_backends()

    def try_open(src, be_list):
        for be in be_list + [0]:  # 0 means AUTO
            cap = cv2.VideoCapture(src, be) if be else cv2.VideoCapture(src)
            if cap.isOpened():
                print(f"[open_smart] Opened {src} with { _name_backend(be) }")
                return cap, src, _name_backend(be)
            tried.append((src, _name_backend(be)))
        return None, None, None

    # 1) Preferred path if exists
    if prefer_path:
        if os.path.exists(prefer_path):
            cap, src, be = try_open(prefer_path, backends)
            if cap: return cap, src, be
        else:
            tried.append((prefer_path, "path-not-found"))

    # 2) Any video under ../data
    any_vid = _find_first_video_in_data()
    if any_vid and (not prefer_path or any_vid != prefer_path):
        cap, src, be = try_open(any_vid, backends)
        if cap: return cap, src, be

    # 3) Webcams 0..2
    for idx in [0, 1, 2]:
        cap, src, be = try_open(idx, [b for b in backends if b != cv2.CAP_FFMPEG])  # FFMPEG not used for cams on Win/Mac typically
        if cap: return cap, src, be

    print("[open_smart] Unable to open any source. Tried:")
    for src, be in tried:
        print(" -", src, "with", be)
    return None, None, None

# ===================== Dashboard helpers =====================
FONT = cv2.FONT_HERSHEY_SIMPLEX

def _txt(img, text, xy, scale=0.6, color=(240,240,240), thick=2):
    cv2.putText(img, text, xy, FONT, scale, color, thick, cv2.LINE_AA)

def _pill(img, x, y, w, h, bg, fg=(255,255,255), label="", value=""):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    if label: _txt(img, label, (x+10, y+int(h*0.65)), 0.55, (220,220,220), 2)
    if value: _txt(img, value, (x+w-10-8*len(value), y+int(h*0.65)), 0.65, fg, 2)

def _barh(img, x, y, w, h, pct, fg=(90,200,255), bg=(55,55,55)):
    pct = max(0.0, min(1.0, float(pct)))
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    cv2.rectangle(img, (x, y), (x+int(w*pct), y+h), fg, -1)

def _sparkline(img, x, y, w, h, values, color=(180,220,255)):
    cv2.rectangle(img, (x, y), (x+w, y+h), (40,40,40), -1)
    if not values: return
    v = np.array(values, dtype=np.float32)
    if np.all(v == v[0]): v = v + 1e-3
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)
    xs = np.linspace(0, w-1, len(v)).astype(int)
    ys = y + h - 1 - (v*(h-2)).astype(int)
    for i in range(1, len(xs)):
        cv2.line(img, (x+xs[i-1], ys[i-1]), (x+xs[i], ys[i]), color, 2)

def make_panel(height):
    panel = np.full((height, PANEL_W, 3), (28,28,30), dtype=np.uint8)
    _txt(panel, "AI Traffic — Live", (18, 32), 0.8, (255,255,255), 2)
    cv2.line(panel, (16, 40), (PANEL_W-16, 40), (60,60,60), 2)
    return panel

def build_dashboard_panel(h, counts_occ, counts_queue, state_adapt, state_fixed, hist_adapt, hist_fixed):
    panel = make_panel(h)
    x = 18; y = 60

    # Active group chips
    a = state_adapt["active_group"]
    f = state_fixed["active_group"]
    _pill(panel, x, y, PANEL_W-36, 34, (0,75,0) if a=="NS" else (0,0,75),
          label="Adaptive Active", value=a); y += 42
    _pill(panel, x, y, PANEL_W-36, 34, (70,60,0) if f=="NS" else (60,50,0),
          label="Fixed Active", value=f); y += 50

    # Green plan bars
    plan = state_adapt["green_plan"]; cyc = max(1e-6, sum(plan.values()))
    _txt(panel, "Planned green (adaptive)", (x, y), 0.62); y += 10
    _txt(panel, f"NS: {plan.get('NS',0):.1f}s", (x, y+22), 0.55, (180,220,255))
    _barh(panel, x+120, y+8, PANEL_W-160, 12, plan.get("NS",0)/cyc, (80,200,120)); y += 26
    _txt(panel, f"EW: {plan.get('EW',0):.1f}s", (x, y+22), 0.55, (180,220,255))
    _barh(panel, x+120, y+8, PANEL_W-160, 12, plan.get("EW",0)/cyc, (80,160,240)); y += 34
    cv2.line(panel, (16, y), (PANEL_W-16, y), (60,60,60), 1); y += 12

    # Per-group occupancy & queue totals
    occ_g = {"NS":0, "EW":0}; q_g = {"NS":0, "EW":0}
    for lid,(g,occ) in counts_occ.items(): occ_g[g] += occ
    for lid,(g,q)   in counts_queue.items(): q_g[g] += q

    _txt(panel, "Demand by group", (x, y), 0.62); y += 10
    for g, col in [("NS",(80,200,120)), ("EW",(80,160,240))]:
        _txt(panel, f"{g}  occ:{occ_g[g]}  q:{q_g[g]}", (x, y+22), 0.57, (220,220,220))
        _barh(panel, x, y+28, PANEL_W-36, 10, min(1.0,(occ_g[g]+2*q_g[g])/10.0), col)  # demo scaling
        y += 34

    cv2.line(panel, (16, y), (PANEL_W-16, y), (60,60,60), 1); y += 12

    # Sparklines (red queue over time)
    _txt(panel, "Red-queue trend (last N)", (x, y), 0.62); y += 6
    _txt(panel, "Adaptive", (x, y+18), 0.55, (200,255,200))
    _sparkline(panel, x+100, y, PANEL_W-132, 24, hist_adapt, (100,220,140)); y += 30
    _txt(panel, "Fixed", (x, y+18), 0.55, (200,220,255))
    _sparkline(panel, x+100, y, PANEL_W-132, 24, hist_fixed, (120,180,255)); y += 34

    cv2.line(panel, (16, y), (PANEL_W-16, y), (60,60,60), 1); y += 12

    # Per-lane table (compact)
    _txt(panel, "Per-lane (occ / q)", (x, y), 0.62); y += 8
    for lid in sorted(counts_occ.keys()):
        grp, occ = counts_occ[lid]
        _grp, q = counts_queue.get(lid, (grp, 0))
        _txt(panel, f"{lid:<12}  {occ:>2} / {q:<2}  [{grp}]", (x, y+22), 0.53, (210,210,210))
        y += 20
        if y > h - 26: break
    return panel

def hstack_right(panel, frame, target_h=None):
    ph = panel.shape[0]
    th = target_h or ph
    fh, fw = frame.shape[:2]
    scale = th / float(fh)
    frame_r = cv2.resize(frame, (int(fw*scale), th), interpolation=cv2.INTER_AREA)
    return np.hstack([panel, frame_r])

# ===================== Drawing helpers =====================
def draw_tracks(img, tracks, lane_asn):
    for t in tracks:
        x,y,w,h = t["bbox"]; cx,cy = t["center"]
        lane_id = lane_asn.get(t["id"], (None,None))[0]
        color = (0,255,0) if lane_id else (0,0,255)
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.circle(img, (cx,cy), 3, (255,255,0), -1)
        label = f'#{t["id"]} {t.get("class","veh")}'
        cv2.putText(img, label, (x, max(0,y-5)), FONT, 0.6, color, 2, cv2.LINE_AA)
        if lane_id:
            cv2.putText(img, lane_id, (x, y+h+18), FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return img

# ===================== Main =====================
def main():
    # Detector, lanes, controllers, tracker, logger
    det = UltralyticsPTDetector("yolov8n.pt", conf=0.35, iou=0.45, imgsz=640, device="")
    lanes = LaneManager(LANES_JSON)
    adapt = AdaptiveSignalController(groups=["NS","EW"], cycle_time=60, min_green=10, max_green=50, smoothing=0.3)
    fixed = FixedSignalController(groups=["NS","EW"], cycle_time=60)
    tracker = SimpleTracker(max_distance=60.0, max_missed=30)
    logger = MetricsLogger(path=str((Path(__file__).resolve().parents[1] / "data" / "metrics.csv").resolve()), groups=("NS","EW"))

    # Rolling history for sparklines
    hist_red_adapt = deque(maxlen=HIST_N)
    hist_red_fixed = deque(maxlen=HIST_N)

    # Open video (file preferred, fallback to webcam)
    cap, used_source, used_backend = open_smart(PREFERRED_VIDEO)
    if cap is None:
        print("Cannot open video/camera. Fix the path or camera permissions and retry.")
        return

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # 1) Detect -> Track
            dets = det.detect(frame)
            tracks = tracker.update(dets)

            # 2) Assign to lanes & queue ROIs, build counts
            counts_occ = {}   # lane_id -> (group, occupancy)
            counts_queue = {} # lane_id -> (group, queue_count)
            lane_asn = {}     # track_id -> (lane_id, group)

            for lane_id, group, _poly, _qpoly in lanes.masks:
                counts_occ[lane_id] = (group, 0)
                counts_queue[lane_id] = (group, 0)

            for t in tracks:
                cx, cy = t["center"]
                lane_id, group = lanes.assign((cx, cy))
                lane_asn[t["id"]] = (lane_id, group)
                if lane_id:
                    g, c = counts_occ[lane_id]
                    counts_occ[lane_id] = (g, c+1)
                    if lanes.in_queue((cx, cy), lane_id):
                        gq, cq = counts_queue[lane_id]
                        counts_queue[lane_id] = (gq, cq+1)
                    # optional per-track bookkeeping
                    if t["current_lane"] is None and lane_id not in t["ever_lanes"]:
                        t["ever_lanes"].add(lane_id)
                        t["current_lane"] = lane_id
                else:
                    t["current_lane"] = None

            # 3) Demand to controller: occupancy + k*queue
            demand = {}
            for lane_id in counts_occ:
                g_occ, occ = counts_occ[lane_id]
                _gq, q = counts_queue[lane_id]
                demand[lane_id] = (g_occ, OCC_ALPHA*occ + QUEUE_BETA*q)

            # 4) Update controllers
            adapt.update(demand)
            state_adapt = adapt.state()  # ensure your traffic_logic has .state()
            fixed.update()
            state_fixed = fixed.state()

            # 5) Logging (per-frame red queue proxies)
            logger.log_frame(
                frame_idx=frame_idx,
                active_adaptive=state_adapt["active_group"],
                active_fixed=state_fixed["active_group"],
                per_lane_occ=counts_occ,
                per_lane_queue=counts_queue,
                lane_to_group=None
            )

            # Compute red-queue this frame for sparklines
            q_ns = sum(q for (grp, q) in counts_queue.values() if grp == "NS")
            q_ew = sum(q for (grp, q) in counts_queue.values() if grp == "EW")
            red_adapt_frame = (q_ew if state_adapt["active_group"] == "NS" else q_ns)
            red_fixed_frame = (q_ew if state_fixed["active_group"] == "NS" else q_ns)
            hist_red_adapt.append(red_adapt_frame)
            hist_red_fixed.append(red_fixed_frame)

            # 6) Build visuals: right video with tracks, left panel with data
            vis_video = draw_tracks(frame.copy(), tracks, lane_asn)
            panel = build_dashboard_panel(
                h=vis_video.shape[0],
                counts_occ=counts_occ,
                counts_queue=counts_queue,
                state_adapt=state_adapt,
                state_fixed=state_fixed,
                hist_adapt=list(hist_red_adapt),
                hist_fixed=list(hist_red_fixed),
            )
            canvas = hstack_right(panel, vis_video)
            cv2.imshow("AI Traffic Management — Dashboard", canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        summary = logger.summary()
        logger.close()
        print("\n=== Metrics Summary ===")
        if summary:
            print(f"Frames observed: {summary['frames']}")
            print(f"Red-queue total (adaptive): {summary['redQ_adaptive_total']:.2f}")
            print(f"Red-queue total (fixed):    {summary['redQ_fixed_total']:.2f}")
            print(f"Improvement vs fixed:       {summary['improvement_%']:.1f}%")
        else:
            print("No frames logged.")
        print(f"Per-frame CSV saved to { (Path(__file__).resolve().parents[1] / 'data' / 'metrics.csv') }")

if __name__ == "__main__":
    main()
