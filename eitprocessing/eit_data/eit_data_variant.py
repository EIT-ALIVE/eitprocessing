"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to when electrical impedance tomographs are read.
"""

import copy
from dataclasses import dataclass
from dataclasses import field
import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self
from eitprocessing.mixins.slicing import SelectByIndex
from eitprocessing.variants import Variant


@dataclass
class EITDataVariant(Variant, SelectByIndex):
    _data_field_name: str = "pixel_impedance"
    pixel_impedance: NDArray = field(repr=False, kw_only=True)

    def __len__(self):
        return self.pixel_impedance.shape[0]

    def __eq__(self, other):
        for attr in ["name", "description", "params"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        if not np.array_equal(
            self.pixel_impedance, other.pixel_impedance, equal_nan=True
        ):
            return False

        return True

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

    def concatenate(self, other: Self) -> Self:
        self.check_equivalence(other, raise_=True)

        return self.__class__(
            label=self.label,
            description=self.description,
            params=self.params,
            pixel_impedance=np.concatenate(
                [self.pixel_impedance, other.pixel_impedance], axis=0
            ),
        )

    def _sliced_copy(
        self, start_index: int, end_index: int, label: str | None = None
    ) -> Self:
        pixel_impedance = self.pixel_impedance[start_index:end_index, :, :]

        return self.__class__(
            label=label,
            description=self.description,
            params=copy.deepcopy(self.params),
            pixel_impedance=pixel_impedance,
        )

    def copy(self, label: str | None = None) -> Self:
        label = label or f"Copy of <{self.label}>"
        return self.__class__(
            label=label,
            description=self.description,
            params=copy.deepcopy(self.params),
            pixel_impedance=np.copy(self.pixel_impedance),
        )
