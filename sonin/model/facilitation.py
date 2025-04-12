from dataclasses import dataclass, field

from sonin.model.gear import Gear


@dataclass
class Facilitation:
    # Larger numbers decrease the degree of each modulation
    granularity: int

    # Larger numbers increase the maximum degrees of modulation
    limit: int

    _gear: Gear = field(init=False)

    def __post_init__(self):
        assert self.granularity >= 1
        assert self.limit >= 0

        self._gear: Gear = Gear(up=self.granularity, down=self.granularity)

    def __call__(self, x: int) -> int:
        return self._gear(x)

    @property
    def current(self) -> int:
        return self._gear.up - self._gear.down

    # Positive values facilitate.
    # Negative values depress.
    def modulate(self, num: int):
        # The granularity is a lower limit on the gear's up and down.
        # Positive modulation decreases down then increases up.
        # Negative modulation decreases up then increases down.
        if self._gear.down == self.granularity:
            up = self._gear.up + num
            self._gear.up = max(self.granularity, min(self.granularity + self.limit, up))

            if up < self.granularity:
                down = self._gear.down + self.granularity - up
                self._gear.down = max(self.granularity, min(self.granularity + self.limit, down))
        else:
            down = self._gear.down - num
            self._gear.down = max(self.granularity, min(self.granularity + self.limit, down))

            if down < self.granularity:
                up = self._gear.up + self.granularity - down
                self._gear.up = max(self.granularity, min(self.granularity + self.limit, up))
