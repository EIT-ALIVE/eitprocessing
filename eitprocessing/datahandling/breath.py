from dataclasses import dataclass


@dataclass
class Breath:
    """Represents a breath with a start, middle and end index."""

    start_time: float
    middle_time: float
    end_time: float

    def __post_init__(self):
        if self.start_time >= self.middle_time or self.middle_time >= self.end_time:
            msg = (
                "Start, middle and end should be consecutive, not "
                "{self.start_time:.2f}, {self.middle_time:.2f} and {self.end_time:.2f}"
            )
            raise ValueError(msg)
