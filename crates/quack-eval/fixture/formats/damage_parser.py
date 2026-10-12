"""Convert storm report damage notation to dollars."""

SUFFIXES = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}


def damage_to_dollars(text: str) -> int:
    """Convert a damage string such as 4.20M to dollars.

    A bare 0 means no measurable damage was recorded.
    """
    text = text.strip().upper()
    if text in {"", "0"}:
        return 0
    suffix = text[-1]
    if suffix in SUFFIXES:
        return int(float(text[:-1]) * SUFFIXES[suffix])
    return int(float(text))
