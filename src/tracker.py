# src/tracker.py
from typing import List, Dict, Tuple
import math
import itertools

class SimpleTracker:
    """
    Lightweight tracker that assigns stable IDs by nearest-center matching.
    - Euclidean matching with a max distance threshold.
    - Tracks age/missed frames; prunes when missed > max_missed.
    - No external deps, good enough for student demos.
    """
    def __init__(self, max_distance: float = 60.0, max_missed: int = 30):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._next_id = 1
        self.tracks = {}  # id -> dict(state)

    def _dist(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        detections: [{"bbox":(x,y,w,h), "center":(cx,cy), "class":str, "conf":float}, ...]
        returns the active tracks in same schema with extra 'id'
        """
        det_centers = [d["center"] for d in detections]
        det_used = [False]*len(detections)

        # 1) match existing tracks -> detections (greedy by distance)
        for tid, t in list(self.tracks.items()):
            t["missed"] += 1
            best_idx, best_d = -1, float("inf")
            for i, c in enumerate(det_centers):
                if det_used[i]: 
                    continue
                d = self._dist(t["center"], c)
                if d < best_d:
                    best_d, best_idx = d, i
            if best_idx != -1 and best_d <= self.max_distance:
                d = detections[best_idx]
                self.tracks[tid].update({
                    "bbox": d["bbox"],
                    "center": d["center"],
                    "class": d.get("class", t.get("class","")),
                    "conf": d.get("conf", t.get("conf", 0.0)),
                    "missed": 0
                })
                det_used[best_idx] = True

        # 2) create new tracks for unmatched detections
        for i, d in enumerate(detections):
            if det_used[i]:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = {
                "id": tid,
                "bbox": d["bbox"],
                "center": d["center"],
                "class": d.get("class",""),
                "conf": d.get("conf", 0.0),
                "missed": 0,
                # bookkeeping for unique lane entries
                "current_lane": None,
                "ever_lanes": set()
            }

        # 3) prune stale tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_missed:
                del self.tracks[tid]

        # return active tracks
        return [self.tracks[k] for k in sorted(self.tracks.keys())]
