# Implementing our own RNG for behavior parity with rust
from typing import Iterable

MASK_4_BYTES = 0xFFFFFFFF
MASK_8_BYTES = 0xFFFFFFFFFFFFFFFF

# https://en.wikipedia.org/wiki/Linear_congruential_generator
LCG_MULTIPLIER = 6364136223846793005


def rotate_right_32(x: int, num: int) -> int:
    x &= MASK_4_BYTES
    return ((x >> num) | (x << (-num & 31))) & MASK_4_BYTES


# https://en.wikipedia.org/wiki/Xorshift
# - faster
# - less random
class XorShift32:
    def __init__(self, seed: int = 1):
        assert seed >= 1

        self.state: int = seed

    def seed(self, new_seed: int):
        assert new_seed >= 1

        self.state = new_seed

    def next_u32(self) -> int:
        x = self.state
        x ^= (x << 13) & MASK_4_BYTES
        x ^= (x >> 17) & MASK_4_BYTES
        x ^= (x << 5) & MASK_4_BYTES
        self.state = x & MASK_4_BYTES
        return self.state


# https://en.wikipedia.org/wiki/Permuted_congruential_generator
# - slower
# - more random
class Pcg32:
    def __init__(self, seed: int = 0, sequence_num: int = 0):
        assert seed >= 0
        assert sequence_num >= 0

        self.increment: int = ((sequence_num << 1) | 1) & MASK_8_BYTES
        self.state: int = (seed + self.increment) & MASK_8_BYTES

        assert self.state > 0, 'if the state is zero, next_u32 fails to produce a large enough random state'

        self.next_u32()

    def seed(self, new_seed: int):
        assert new_seed >= 0

        self.state = (new_seed + self.increment) & MASK_8_BYTES

        assert self.state > 0, 'if the state is zero, next_u32 fails to produce a large enough random state'

        self.next_u32()

    def next_u32(self) -> int:
        x = self.state
        rotate_num = x >> 59
        self.state = (x * LCG_MULTIPLIER + self.increment) & MASK_8_BYTES
        x ^= x >> 18
        return rotate_right_32(x >> 27, rotate_num)


rng = Pcg32()  # changing this implementation will affect tests
seed = rng.seed
rand_u32 = rng.next_u32


def rand_bool() -> bool:
    if rand_u32() & 1:
        return True
    else:
        return False


def rand_int(lower: int | None = None, upper: int | None = None) -> int:
    assert lower is None or (-(2 ** 32) <= lower <= 2 ** 32 - 1)
    assert upper is None or (-(2 ** 32) <= upper <= 2 ** 32 - 1)
    assert lower is None or upper is None or lower <= upper

    if lower is None and upper is None:
        return rand_u32()
    elif lower is not None and upper is None:
        upper = lower
        lower = 0
    elif lower is None:
        lower = 0

    return (rand_u32() % (1 + upper - lower)) + lower


def rand_sign() -> int:
    if rand_bool():
        return 1
    else:
        return -1


def choice[T](items: Iterable[T]) -> T:
    as_tuple = tuple(items)
    return as_tuple[rand_int(0, len(as_tuple) - 1)]


def shuffle(items: list):
    for i in range(len(items)):
        j = rand_int(0, len(items) - 1)
        temp = items[i]
        items[i] = items[j]
        items[j] = temp
