from pydantic import BaseModel, Field

from sonin.sonin_math import div


class Gear(BaseModel):
    # Increases output speed
    up: int = Field(default=1, ge=1)

    # Decreases output speed
    down: int = Field(default=1, ge=1)

    # Stores the progress towards the next output to avoid loss of precision due to int division rounding.
    # Changes to up and down should scale the current as well, but this is likely an unnecessary level of precision.
    current: int = 0

    def __call__(self, x: int) -> int:
        total = self.current + x * self.up
        self.current = total % self.down
        return div(total, self.down)
