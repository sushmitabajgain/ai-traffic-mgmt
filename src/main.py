import cv2
from ultralytics_detector import UltralyticsPTDetector
from lane_counter import LaneManager
from traffic_logic import AdaptiveSignalController

VIDEO = 0   # use 0 for webcam
# VIDEO = "../data/sample.mp4"
LANES = "../lanes/lanes.json"

def draw_dets(img, dets, lane_asn):
    for d in dets:
        x,y,w,h = d["bbox"]; cx,cy = d["center"]
        lane_id = lane_asn.get((cx,cy), (None,None))[0]
        color = (0,255,0) if lane_id else (0,0,255)
        cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        cv2.circle(img, (cx,cy), 3, (255,255,0), -1)
        cv2.putText(img, f'{d["class"]} {d["conf"]:.2f}', (x, max(0,y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if lane_id: cv2.putText(img, lane_id, (x, y+h+18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img

def draw_overlay(img, lanes, counts, state):
    out = lanes.draw(img)
    # left panel: counts
    y0 = 30
    for i, (lid, (grp, c)) in enumerate(counts.items()):
        cv2.putText(out, f"{lid}: {c}", (20, y0 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    # right panel: controller
    h,w = out.shape[:2]; x0 = w-330
    cv2.rectangle(out, (x0-10,10), (w-10,165), (0,0,0), -1)
    cv2.putText(out, "Signal Controller", (x0,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(out, f"Active: {state['active_group']}", (x0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(out, f"Remaining: {state['phase_remaining']:.1f}s", (x0,85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    y = 110
    for g,v in state["green_plan"].items():
        cv2.putText(out, f"{g}: {v:.1f}s  dens~{state['density_ema'][g]:.1f}", (x0,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        y += 22
    # tiny lights (top-left)
    active = state["active_group"]
    ns_color = (0,255,0) if active=="NS" else (0,0,255)
    ew_color = (0,255,0) if active=="EW" else (0,0,255)
    cv2.circle(out, (50,50), 12, ns_color, -1); cv2.putText(out, "NS", (70,56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.circle(out, (50,80), 12, ew_color, -1); cv2.putText(out, "EW", (70,86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return out

def main():
    det = UltralyticsPTDetector("yolov8n.pt", conf=0.35, iou=0.45, imgsz=640, device="")
    lanes = LaneManager(LANES)
    ctrl = AdaptiveSignalController(groups=["NS","EW"], cycle_time=60, min_green=10, max_green=50, smoothing=0.3)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened(): print("Cannot open video/camera"); return

    while True:
        ok, frame = cap.read()
        if not ok: break

        dets = det.detect(frame)

        # init counts
        counts = {}
        for lane_id, group, _ in lanes.masks:
            counts[lane_id] = (group, 0)

        # assign and count
        lane_asn = {}
        for d in dets:
            cx, cy = d["center"]
            lane_id, group = lanes.assign((cx, cy))
            lane_asn[(cx,cy)] = (lane_id, group)
            if lane_id:
                g,c = counts[lane_id]
                counts[lane_id] = (g, c+1)

        # update controller & draw
        ctrl.update(counts)
        state = ctrl.state()
        vis = draw_dets(frame.copy(), dets, lane_asn)
        vis = draw_overlay(vis, lanes, counts, state)

        cv2.imshow("AI Traffic Management (Ultralytics YOLO)", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
