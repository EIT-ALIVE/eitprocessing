from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    """Single time point event registered during an EIT measurement."""

    marker: int
    text: str
