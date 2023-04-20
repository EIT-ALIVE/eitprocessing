from dataclasses import dataclass, field


@dataclass
class Event:
    index: int
    marker: int = field(repr=False)
    text: str
