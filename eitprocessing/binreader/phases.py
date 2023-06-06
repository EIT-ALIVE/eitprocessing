"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to when electrical impedance tomographs are read.
"""

from dataclasses import dataclass
from dataclasses import field


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


@dataclass
class QRSMark(PhaseIndicator):
    pass
