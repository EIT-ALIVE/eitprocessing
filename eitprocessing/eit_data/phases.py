from dataclasses import dataclass


@dataclass
class PhaseIndicator:
    index: int
    time: float


@dataclass
class MinValue(PhaseIndicator):
    pass


@dataclass
class MaxValue(PhaseIndicator):
    pass


@dataclass
class QRSMark(PhaseIndicator):
    pass
