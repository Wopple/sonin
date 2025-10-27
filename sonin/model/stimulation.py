from typing import Any

from pydantic import BaseModel, Field

from sonin.sonin_math import div


class SnapBack(BaseModel):
    # The starting value and the value it snaps back to.
    baseline: int = 0

    # Higher values create a faster snap back
    # 1: no snap back adjustment
    restore_rate: int = Field(default=1, ge=1)

    # Higher values create a slower snap back
    # 0: snaps back all the way immediately
    restore_damper: int = Field(default=0, ge=0)

    _value: int = None

    def model_post_init(self, context: Any, /):
        # value == 0 means it is at baseline.
        # Baseline adjustments happen transparently at get and set time.
        # The step math is easier with a virtual baseline of zero.
        self.value: int = 0

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


class Stimulation(BaseModel):
    # Value increment on each stimulation
    amount: int = Field(default=1, ge=1)

    snap_back: SnapBack = Field(default_factory=SnapBack)

    @property
    def value(self) -> int:
        return self.snap_back.value

    @value.setter
    def value(self, value):
        self.snap_back.value = value

    def step(self):
        self.snap_back.step()

    def stimulate(self):
        self.snap_back._value += self.amount
