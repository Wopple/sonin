from random import randint


def rand_sign() -> int:
    if randint(0, 1) == 0:
        return -1
    else:
        return 1
