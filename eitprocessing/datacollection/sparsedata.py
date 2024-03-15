from dataclasses import dataclass


class SparseData:
    """SparseData."""


@dataclass
class Event(SparseData):
    """Single time point event registered during an EIT measurement."""

    index: int
    time: float
    marker: int
    text: str


@dataclass
class PhaseIndicator(SparseData):
    """Parent class for phase indications."""

    index: int
    time: float


@dataclass
class MinValue(PhaseIndicator):
    """Automatically registered local minimum of an EIT measurement."""


@dataclass
class MaxValue(PhaseIndicator):
    """Automatically registered local maximum of an EIT measurement."""


@dataclass
class QRSMark(PhaseIndicator):
    """Automatically registered QRS mark an EIT measurement from a Timpel device."""
