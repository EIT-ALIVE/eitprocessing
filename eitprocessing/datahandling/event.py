from dataclasses import dataclass


@dataclass
class Event:
    """Single time point event registered during an EIT measurement."""

    marker: int
    text: str
