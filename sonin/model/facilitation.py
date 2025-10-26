from typing import Any

from pydantic import BaseModel, Field

from sonin.model.gear import Gear


class Facilitation(BaseModel):
    """
    Facilitation scales an input up or down.
    It can be modulated to either increase or decrease the scaling.
    """

    # Larger numbers decrease the degree of each modulation.
    # This works by setting a minimum bound on a gear's values.
    # With a larger value, each increment of a gear's value is less impactful allowing for finer grained facilitation.
    granularity: int = Field(ge=1)

    # The maximum degrees of facilitation.
    # The impact of each degree is determined by the granularity.
    limit: int = Field(ge=0)

    gear: Gear = None

    def model_post_init(self, context: Any, /):
        assert self.gear is None

        self.gear = Gear(up=self.granularity, down=self.granularity)

    def __call__(self, x: int) -> int:
        return self.gear(x)

    @property
    def current(self) -> int:
        return self.gear.up - self.gear.down

    def modulate(self, num: int):
        """
        Positive values for num cause more facilitation.
        Negative values for num cause more depression.
        """

        # Apply the modulation to up if down is already at the minimum.
        if self.gear.down == self.granularity:
            up = self.gear.up + num
            self.gear.up = max(self.granularity, min(self.granularity + self.limit, up))

            # Apply the delta below the minimum up value to increase the down value if any.
            if up < self.granularity:
                down = self.gear.down + self.granularity - up
                self.gear.down = max(self.granularity, min(self.granularity + self.limit, down))
        # Otherwise apply the modulation to down.
        else:
            down = self.gear.down - num
            self.gear.down = max(self.granularity, min(self.granularity + self.limit, down))

            # Apply the delta below the minimum down value to increase the up value if any.
            if down < self.granularity:
                up = self.gear.up + self.granularity - down
                self.gear.up = max(self.granularity, min(self.granularity + self.limit, up))
