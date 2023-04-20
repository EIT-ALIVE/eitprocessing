from dataclasses import dataclass


@dataclass
class TimingError:
    index: int
    time: float
    timing_error: int
