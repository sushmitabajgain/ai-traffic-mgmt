import json, numpy as np, cv2

class LaneManager:
    def __init__(self, lanes_json):
        data = json.load(open(lanes_json, "r"))
        self.masks = [(lane["id"], lane["group"], np.array(lane["polygon"], dtype=np.int32)) for lane in data["lanes"]]

    def assign(self, center_xy):
        for lane_id, group, poly in self.masks:
            if cv2.pointPolygonTest(poly, center_xy, False) >= 0:
                return lane_id, group
        return None, None

    def draw(self, frame):
        out = frame.copy()
        for lane_id, group, poly in self.masks:
            cv2.polylines(out, [poly], True, (255,255,255), 2)
            x,y = poly[0]
            cv2.putText(out, f"{lane_id}({group})", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return out
