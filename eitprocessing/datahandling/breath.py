from typing import NamedTuple


class Breath(NamedTuple):
    """Represents a breath with a start, middle and end index."""

    start_index: int
    middle_index: int
    end_index: int
