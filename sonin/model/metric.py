# Record behavior to make functional decisions and improve developer insight
from sonin.sonin_math import div


class FrequencyProfile:
    def __init__(self, size: int):
        assert size >= 1

        self.size: int = size
        self.last_time: int = -1
        self.deltas: list[int] = []
        self.i_next: int = 0

    def record(self, c_time: int):
        if self.last_time == -1:
            self.last_time = c_time
            return

        if len(self.deltas) == self.size:
            self.deltas[self.i_next] = c_time - self.last_time
            self.i_next = (self.i_next + 1) % self.size
        else:
            self.deltas.append(c_time - self.last_time)

        self.last_time = c_time

    def is_full(self) -> bool:
        return len(self.deltas) == self.size

    @property
    def mean(self) -> int:
        if len(self.deltas) == 0:
            return -1

        return div(sum(self.deltas), len(self.deltas))

    @property
    def instability(self) -> int:
        """
        Sum of distances from the mean of all recorded deltas.
        """
        mean = self.mean

        if mean == -1:
            return -1

        return sum(abs(d - mean) for d in self.deltas)
