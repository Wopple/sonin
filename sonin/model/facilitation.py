from sonin.model.gear import Gear

class Facilitation:
    def __init__(self, granularity: int, limit: int):
        assert granularity >= 1
        assert limit >= 0

        self.granularity: int = granularity
        self.limit: int = limit
        self.gear: Gear = Gear(granularity, granularity)

    def __call__(self, x: int) -> int:
        return self.gear(x)

    @property
    def current(self) -> int:
        return self.gear.up - self.gear.down

    def modulate(self, num: int):
        # The granularity is a lower limit on the gear.
        # Positive modulation decreases down then increases up.
        # Negative modulation decreases up then increases down.
        if self.gear.down == self.granularity:
            up = self.gear.up + num
            self.gear.up = max(self.granularity, min(self.granularity + self.limit, up))

            if up < self.granularity:
                down = self.gear.down + self.granularity - up
                self.gear.down = max(self.granularity, min(self.granularity + self.limit, down))
        else:
            down = self.gear.down - num
            self.gear.down = max(self.granularity, min(self.granularity + self.limit, down))

            if down < self.granularity:
                up = self.gear.up + self.granularity - down
                self.gear.up = max(self.granularity, min(self.granularity + self.limit, up))
