"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to when electrical impedance tomographs are read.
"""

import copy
from dataclasses import dataclass
from dataclasses import field
import numpy as np
from typing_extensions import Self
from ..variants import Variant


@dataclass
class EITDataVariant(Variant):
    pixel_impedance: np.ndarray = field(repr=False, kw_only=True)

    def __len__(self):
        return self.pixel_impedance.shape[0]

    def __eq__(self, other):
        for attr in ["name", "description", "params"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        for attr in ["pixel_values"]:
            # NaN values are not equal. Check whether values are equal or both NaN.
            s = getattr(self, attr)
            o = getattr(other, attr)
            if not np.all((s == o) | (np.isnan(s) & np.isnan(o))):
                return False

        return True

    def select_by_indices(self, indices):
        obj = self.deepcopy()
        obj.pixel_impedance = self.pixel_impedance[indices, :, :]
        return obj

    __getitem__ = select_by_indices

    @property
    def global_baseline(self):
        return np.nanmin(self.pixel_impedance)

    @property
    def pixel_impedance_global_offset(self):
        return self.pixel_impedance - self.global_baseline

    @property
    def pixel_baseline(self):
        return np.nanmin(self.pixel_impedance, axis=0)

    @property
    def pixel_impedance_individual_offset(self):
        return self.pixel_impedance - np.min(self.pixel_impedance, axis=0)

    @property
    def global_impedance(self):
        return np.nansum(self.pixel_impedance, axis=(1, 2))

    @classmethod
    def concatenate(cls, a: Self, b: Self) -> Self:
        cls.check_equivalence(a, b, raise_=True)

        return cls(
            name=a.name,
            description=a.description,
            params=a.params,
            pixel_impedance=np.concatenate(
                [a.pixel_impedance, b.pixel_impedance], axis=0
            ),
        )

    deepcopy = copy.deepcopy
