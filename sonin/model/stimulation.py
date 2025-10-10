from dataclasses import dataclass, field

from sonin.sonin_math import div


@dataclass
class SnapBack:
    # The starting value and the value it snaps back to.
    baseline: int = 0

    # Higher values create a faster snap back
    # 1: no snap back adjustment
    restore_rate: int = 1

    # Higher values create a slower snap back
    # 0: snaps back all the way immediately
    restore_damper: int = 0

    _value: int = field(init=False)

    def __post_init__(self):
        # _value == 0 means it is at baseline.
        # Baseline adjustments happen transparently at get and set time.
        # The step math is easier with a virtual baseline of zero.
        self._value: int = 0

        assert self.restore_rate >= 1
        assert self.restore_damper >= 0
        assert self.restore_rate >= self.restore_damper

    @property
    def value(self) -> int:
        return self._value + self.baseline

    @value.setter
    def value(self, value):
        self._value = value - self.baseline

    def step(self):
        if self._value != 0:
            self._value = div(self._value * self.restore_damper, self.restore_rate)
