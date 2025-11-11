from enum import IntEnum

from pydantic import BaseModel

from sonin.model.gear import Gear

identity = Gear(up=1, down=1)


class Lesson(IntEnum):
    """
    Enum names depict what a mind needs.

    Example:
        MORE_AXON_MOVEMENT means the DNA should mutate in ways to make the axons move more.
    """

    MORE_AXON_MOVEMENT = 0
    LESS_AXON_MOVEMENT = 0


class LessonPlan(BaseModel):
    """
    Producer: Coaches determine what the mind needs.
    Consumer: Mutagens interpret those needs to modify the DNA.
    """

    plan: dict[Lesson, Gear]

    def __getitem__(self, item: Lesson) -> Gear:
        return self.plan.get(item, identity)
