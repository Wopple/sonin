class SnapBack:
    def __init__(self, baseline: int = 0, restore_rate: int = 1, restore_scalar: int = 1):
        # Baseline adjustments happen transparently at get and set time. The step
        # math is easier with a baseline of zero.
        self.baseline: int = baseline
        self._value: int = 0

        self.restore_rate: int = restore_rate
        self.restore_scalar: int = restore_scalar

    @property
    def value(self) -> int:
        return self._value + self.baseline

    @value.setter
    def value(self, value):
        self._value = value - self.baseline

    def step(self):
        if self._value > 0:
            self._value = self._value * self.restore_scalar // self.restore_rate
        elif self._value < 0:
            positive = -self._value
            self._value = -(positive * self.restore_scalar // self.restore_rate)
