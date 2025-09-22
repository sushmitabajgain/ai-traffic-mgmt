import time

class AdaptiveSignalController:
    def __init__(self, groups, cycle_time=60, min_green=10, max_green=50, smoothing=0.3):
        self.groups = groups
        self.cycle, self.min_g, self.max_g, self.alpha = cycle_time, min_green, max_green, smoothing
        self.dens = {g: 1.0 for g in groups}
        self.plan = {g: cycle_time/len(groups) for g in groups}
        self.idx, self.t0 = 0, time.time()
        self.remaining = self.plan[self.groups[0]]

    def _replan(self):
        total = sum(self.dens.values())
        if total <= 0:
            eq = self.cycle/len(self.groups)
            self.plan = {g: max(self.min_g, min(self.max_g, eq)) for g in self.groups}
            return
        raw = {g: max(self.min_g, min(self.max_g, self.cycle*(self.dens[g]/total))) for g in self.groups}
        s = sum(raw.values())
        self.plan = {g: v*(self.cycle/s) for g, v in raw.items()} if s>0 else {g: self.cycle/len(self.groups) for g in self.groups}

    def update(self, per_lane_counts):
        # sum per group
        pg = {g: 0 for g in self.groups}
        for _lane, (g, c) in per_lane_counts.items():
            if g in pg: pg[g] += c
        # EMA
        for g in self.groups:
            self.dens[g] = (1-self.alpha)*self.dens[g] + self.alpha*pg[g]
        # phase timing
        now = time.time()
        elapsed = now - self.t0
        left = self.plan[self.groups[self.idx]] - elapsed
        if left <= 0:
            self.idx = (self.idx + 1) % len(self.groups)
            if self.idx == 0: self._replan()
            self.t0 = now
            left = self.plan[self.groups[self.idx]]
        self.remaining = max(0.0, left)

    def state(self):
        return {"active_group": self.groups[self.idx], "green_plan": self.plan, "phase_remaining": self.remaining, "density_ema": self.dens}
