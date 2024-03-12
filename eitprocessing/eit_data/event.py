from dataclasses import dataclass


@dataclass
class Event:
    """Single time point event registered during an EIT measurement."""

    index: int
    time: float
    marker: int
    text: str
