from dataclasses import dataclass, field


@dataclass
class PhaseIndicator:
    index: int
    time: float = field(repr=False)


@dataclass
class MinValue(PhaseIndicator):
    pass


@dataclass
class MaxValue(PhaseIndicator):
    pass
