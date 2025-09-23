# src/metrics_logger.py
import csv, os
from collections import defaultdict

class MetricsLogger:
    """
    Logs per-frame queue metrics and keeps running totals for a simple 'red queue time' proxy.
    Also logs a fixed-timer baseline in parallel.
    """
    def __init__(self, path="../data/metrics.csv", groups=("NS","EW")):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        header = ["frame", "active_adaptive", "active_fixed"]
        for g in groups:
            header += [f"q_{g}", f"occ_{g}"]
        header += ["redQ_adaptive_frame", "redQ_fixed_frame"]
        self.w.writerow(header)
        self.groups = list(groups)
        self.total_redQ_adaptive = 0.0
        self.total_redQ_fixed = 0.0
        self.frames = 0

    def log_frame(self, frame_idx, active_adaptive, active_fixed, per_lane_occ, per_lane_queue, lane_to_group):
        # sum per-group occupancy and queue
        occ_g = {g:0 for g in self.groups}
        q_g = {g:0 for g in self.groups}
        for lane_id, (grp, occ) in per_lane_occ.items():
            occ_g[grp] += occ
        for lane_id, (grp, q) in per_lane_queue.items():
            q_g[grp] += q

        # "red queue" this frame = sum of queue in groups that are NOT active
        red_adapt = sum(v for g,v in q_g.items() if g != active_adaptive)
        red_fixed = sum(v for g,v in q_g.items() if g != active_fixed)
        self.total_redQ_adaptive += red_adapt
        self.total_redQ_fixed += red_fixed
        self.frames += 1

        row = [frame_idx, active_adaptive, active_fixed]
        for g in self.groups:
            row += [q_g[g], occ_g[g]]
        row += [red_adapt, red_fixed]
        self.w.writerow(row)

    def summary(self):
        if self.frames == 0:
            return {}
        return {
            "frames": self.frames,
            "redQ_adaptive_total": self.total_redQ_adaptive,
            "redQ_fixed_total": self.total_redQ_fixed,
            "improvement_%": ( (self.total_redQ_fixed - self.total_redQ_adaptive) / max(1e-6, self.total_redQ_fixed) ) * 100.0
        }

    def close(self):
        self.f.flush()
        self.f.close()
