class Gear:
    def __init__(self, up: int = 1, down: int = 1):
        assert up >= 1
        assert down >= 1

        self.up: int = up
        self.down: int = down
        self.current: int = 0

    def __call__(self, x: int) -> int:
        total = self.current + x * self.up
        self.current = total % self.down
        return total // self.down
