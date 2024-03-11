from dataclasses import dataclass


@dataclass
class Event:
    index: int
    time: float
    marker: int
    text: str
