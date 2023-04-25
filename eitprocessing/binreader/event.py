"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods to deal with events when electrical impedance tomographs are read.
"""


from dataclasses import dataclass
from dataclasses import field


@dataclass
class Event:
    index: int
    marker: int = field(repr=False)
    text: str
