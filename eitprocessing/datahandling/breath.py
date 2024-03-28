from typing import NamedTuple


class Breath(NamedTuple):
    """Represents a breath with a start, middle and end index."""

    start_time: float
    middle_time: float
    end_time: float
