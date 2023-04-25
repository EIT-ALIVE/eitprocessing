"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to timing errors when electrical impedance tomographs are read.
"""

from dataclasses import dataclass


@dataclass
class TimingError:
    index: int
    time: float
    timing_error: int
