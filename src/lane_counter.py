# src/lane_counter.py
import json, numpy as np, cv2

class LaneManager:
    def __init__(self, lanes_json):
        data = json.load(open(lanes_json, "r"))
        self.masks = []  # (lane_id, group, poly, queue_poly or None)
        for lane in data["lanes"]:
            poly = np.array(lane["polygon"], dtype=np.int32)
            qpoly = np.array(lane["queue_polygon"], dtype=np.int32) if "queue_polygon" in lane else None
            self.masks.append((lane["id"], lane["group"], poly, qpoly))

    def assign(self, center_xy):
        for lane_id, group, poly, _ in self.masks:
            if cv2.pointPolygonTest(poly, center_xy, False) >= 0:
                return lane_id, group
        return None, None

    def in_queue(self, center_xy, lane_id):
        # True if center lies inside the lane's queue polygon (if any)
        for lid, _g, _p, qpoly in self.masks:
            if lid == lane_id and qpoly is not None:
                return cv2.pointPolygonTest(qpoly, center_xy, False) >= 0
        return False

    def draw(self, frame):
        out = frame.copy()
        # draw lanes
        for lane_id, group, poly, qpoly in self.masks:
            cv2.polylines(out, [poly], True, (255,255,255), 2)
            x,y = poly[0]
            cv2.putText(out, f"{lane_id}({group})", (int(x), int(y)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            # draw queue polygon (yellow)
            if qpoly is not None:
                cv2.polylines(out, [qpoly], True, (0,255,255), 2)
        return out
