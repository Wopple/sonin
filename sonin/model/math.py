def div(a: int, b: int) -> int:
    """
    Python rounds to -inf. This is wrong for this project. Rounds to zero instead.
    """
    if (a < 0) != (b < 0) and a % b != 0:
        return a // b + 1
    else:
        return a // b
