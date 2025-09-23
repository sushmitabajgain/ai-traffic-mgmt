# src/fixed_controller.py
import time

class FixedSignalController:
    """
    Simple round-robin controller with fixed green splits.
    Used in parallel as a baseline for metrics.
    """
    def __init__(self, groups, cycle_time=60, plan=None):
        self.groups = groups
        self.cycle_time = cycle_time
        if plan is None:
            per = cycle_time / len(groups)
            self.plan = {g: per for g in groups}
        else:
            self.plan = plan
        self.idx = 0
        self.t0 = time.time()
        self.remaining = self.plan[self.groups[0]]

    def update(self):
        now = time.time()
        elapsed = now - self.t0
        left = self.plan[self.groups[self.idx]] - elapsed
        if left <= 0:
            self.idx = (self.idx + 1) % len(self.groups)
            self.t0 = now
            left = self.plan[self.groups[self.idx]]
        self.remaining = max(0.0, left)

    def state(self):
        return {"active_group": self.groups[self.idx],
                "green_plan": self.plan,
                "phase_remaining": self.remaining}
