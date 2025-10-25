def div(a: int, b: int) -> int:
    """
    Python rounds towards -inf. This is wrong for this project. This rounds towards zero instead.
    Rust rounds towards zero already, so this is necessary for parity of behavior.
    """
    result = a // b
    is_negative = result < 0
    is_rounded = a % b != 0

    if is_negative and is_rounded:
        # round the other way
        return result + 1
    else:
        return result
