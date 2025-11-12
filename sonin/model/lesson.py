from enum import IntEnum
from math import gcd

from pydantic import BaseModel

from sonin.model.gear import Gear
from sonin.sonin_math import div

identity = Gear(up=1, down=1)


class Lesson(IntEnum):
    """
    Enum names depict what a mind needs.

    Example:
        MORE_ACTIVATION means the DNA should mutate in ways to make more activations.
    """

    MORE_ACTIVATION = 0
    LESS_ACTIVATION = 1
    MORE_AXON_MOVEMENT = 2
    LESS_AXON_MOVEMENT = 3


class LessonPlan(BaseModel):
    """
    Producer: Coaches determine what the mind needs.
    Consumer: Mutagens interpret those needs to modify the DNA.
    """

    plan: dict[Lesson, Gear]

    def __getitem__(self, item: Lesson | tuple[Lesson, ...]) -> Gear:
        if isinstance(item, Lesson):
            return self.plan.get(item, identity)
        elif isinstance(item, tuple):
            up = 1
            down = 1

            for lesson in item:
                gear = self.plan.get(lesson, identity)
                up *= gear.up
                down *= gear.down

            divisor = gcd(up, down)
            return Gear(up=div(up, divisor), down=div(down, divisor))
