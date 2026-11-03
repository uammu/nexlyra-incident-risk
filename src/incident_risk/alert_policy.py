import time

LEVELS = {0: "none", 1: "review", 2: "acknowledge", 3: "page"}


def alert_level(p: float, thr: float, environment: str, policy: dict) -> int:
    """0 none | 1 review (ticket) | 2 acknowledge (on-call, 15 min) | 3 page. Only listed environments page."""
    if p < thr:
        return 0
    level = 1
    if p >= min(thr * policy["l2_multiplier"], 0.95):
        level = 2
    if p >= min(thr * policy["l3_multiplier"], 0.95):
        level = 3
    if level == 3 and environment not in policy["page_environments"]:
        level = 2
    return level


class Suppressor:
    """Mute repeats for the same (service, region, environment) inside a cooldown; escalations always pass."""

    def __init__(self, cooldown_minutes: float):
        self.cool, self.last = cooldown_minutes * 60, {}

    def check(self, key, level: int, now: float | None = None) -> bool:
        if level == 0:
            return False
        now = time.time() if now is None else now
        prev = self.last.get(key)
        if prev and now - prev[0] < self.cool and level <= prev[1]:
            return True
        self.last[key] = (now, level)
        return False
