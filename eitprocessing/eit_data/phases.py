from dataclasses import dataclass


@dataclass
class PhaseIndicator:
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
