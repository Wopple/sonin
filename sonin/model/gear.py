from dataclasses import dataclass


@dataclass
class Gear:
    # Increases output speed
    up: int

    # Decreases output speed
    down: int

    # Stores the progress towards the next output
    current: int = 0

    def __call__(self, x: int) -> int:
        total = self.current + x * self.up
        self.current = total % self.down
        return total // self.down
