from dataclasses import dataclass, field

from sonin.model.math import div


@dataclass
class SnapBack:
    # The starting value and the value it snaps back to.
    baseline: int = 0

    # Higher values create a faster snap back
    # 1: no snap back adjustment
    restore_rate: int = 1

    # Higher values create a slower snap back
    # 0: snaps back all the way immediately
    restore_scalar: int = 0

    _value: int = field(init=False)

    def __post_init__(self):
        # _value == 0 means it is at baseline.
        # Baseline adjustments happen transparently at get and set time.
        # The step math is easier with a virtual baseline of zero.
        self._value: int = 0

        assert self.restore_rate >= 1
        assert self.restore_scalar >= 0
        assert self.restore_rate >= self.restore_scalar

    @property
    def value(self) -> int:
        return self._value + self.baseline

    @value.setter
    def value(self, value):
        self._value = value - self.baseline

    def step(self):
        if self._value > 0:
            self._value = div(self._value * self.restore_scalar, self.restore_rate)
        elif self._value < 0:
            positive = -self._value
            self._value = -(div(positive * self.restore_scalar, self.restore_rate))
