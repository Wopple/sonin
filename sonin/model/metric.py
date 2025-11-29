# Record behavior to make functional decisions and improve developer insight
from pydantic import BaseModel, Field

from sonin.sonin_math import div


class Metric(BaseModel):
    """
    Basic metrics over a collection of values.
    """

    values: list[int] = Field(default_factory=list)

    def record(self, value: int):
        self.values.append(value)

    @property
    def size(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> int | None:
        if self.values:
            return div(sum(self.values), len(self.values))
        else:
            return None

    @property
    def instability(self) -> int | None:
        """
        Sum of distances from the mean of all recorded values.
        """

        mean = self.mean

        if mean is not None:
            return sum(abs(d - mean) for d in self.values)
        else:
            return None


class SlidingFrequencyProfile(BaseModel):
    """
    Records a sliding window of the elapsed time since the last record based on the c_time of the events. Calculates
    metrics related to the frequency of events.
    """

    size: int = Field(ge=1)
    last_time: int = -1
    deltas: list[int] = Field(default_factory=list)
    i_next: int = 0

    def record(self, c_time: int):
        if self.last_time == -1:
            self.last_time = c_time
            return

        if self.is_full():
            # the buffer has filled, treat like a ring buffer
            self.deltas[self.i_next] = c_time - self.last_time
            self.i_next = (self.i_next + 1) % self.size
        else:
            # continue to add to the buffer
            self.deltas.append(c_time - self.last_time)

        self.last_time = c_time

    def is_full(self) -> bool:
        return len(self.deltas) == self.size

    @property
    def mean(self) -> int | None:
        if self.deltas:
            return div(sum(self.deltas), len(self.deltas))
        else:
            return None

    @property
    def instability(self) -> int | None:
        """
        Sum of distances from the mean of all recorded deltas.
        """

        mean = self.mean

        if mean is not None:
            return sum(abs(d - mean) for d in self.deltas)
        else:
            return None
