import bisect
import itertools
import warnings
from dataclasses import dataclass
from typing import Literal
import numpy as np
from . import ROISelection


@dataclass
class GridSelection(ROISelection):
    v_split: int
    h_split: int
    split_pixels: bool = False

    def __post_init__(self):
        if not isinstance(self.v_split, int):
            raise TypeError(
                "Invalid type for `h_split`. "
                f"Should be `int`, not {type(self.h_split)}."
            )

        if not isinstance(self.h_split, int):
            raise TypeError(
                "Invalid type for `h_split`. "
                f"Should be `int`, not {type(self.h_split)}."
            )

        if self.v_split < 1:
            raise InvalidVerticalDivision("`v_split` can't be smaller than 1.")

        if self.h_split < 1:
            raise InvalidHorizontalDivision("`h_split` can't be smaller than 1.")

        if not isinstance(self.split_pixels, bool):
            raise TypeError(
                "Invalid type for `split_pixels`. "
                f"Should be `bool`, not {type(self.split_pixels)}"
            )

        if self.split_pixels is True:
            raise NotImplementedError(
                "GrisSelection has no support for split pixels yet."
            )

    def find_grid(self, data) -> list:
        n_rows = data.shape[0]
        n_columns = data.shape[1]
        if self.h_split > n_columns:
            raise InvalidHorizontalDivision(
                f"`h_split` ({self.h_split}) is larger than the number of columns ({n_columns})."
            )

        if self.v_split > n_rows:
            raise InvalidVerticalDivision(
                f"`v_split` ({self.v_split}) is larger than the number or rows ({n_rows})."
            )

        if self.split_pixels:
            return self._find_grid_split_pixels(data)

        return self._find_grid_no_split_pixels(data)

    def _find_grid_no_split_pixels(self, data):
        # Detect upper, lower, left and right column of grid selection,
        # i.e, find the first row and column where the sum of pixels in that column is not NaN

        is_numeric = ~np.isnan(data)

        def get_region_boundaries(
            orientation: Literal["horizontal", "vertical"], n_regions: int
        ):
            horizontal = orientation == "horizontal"
            axis = 0 if horizontal else 1
            vector_has_numeric_cells = is_numeric.sum(axis) > 0
            numeric_vector_indices = np.argwhere(vector_has_numeric_cells)
            first_numeric_vector = numeric_vector_indices.min()
            last_vector_numeric = numeric_vector_indices.max()

            n_vectors = last_vector_numeric - first_numeric_vector + 1
            n_vectors_per_region = n_vectors / n_regions

            if n_vectors_per_region % 1 > 0:
                warnings.warn(
                    f"The {orientation} groups will not have an equal number of {'columns' if horizontal else 'rows'}. "
                    f"{n_vectors} is not equally divisible by {n_regions}.",
                    RuntimeWarning,
                )

            region_boundaries = [
                first_numeric_vector
                + bisect.bisect_left(np.arange(n_vectors) / n_vectors_per_region, c)
                for c in range(n_regions + 1)
            ]
            return region_boundaries

        h_boundaries = get_region_boundaries("horizontal", n_regions=self.h_split)
        v_boundaries = get_region_boundaries("vertical", n_regions=self.v_split)

        matrices = []
        for v_start, v_end in itertools.pairwise(v_boundaries):
            for h_start, h_end in itertools.pairwise(h_boundaries):
                matrix = np.copy(is_numeric)
                matrix[:, :h_start] = False
                matrix[:, h_end:] = False
                matrix[:v_start, :] = False
                matrix[v_end:, :] = False
                matrices.append(matrix)

        return matrices

    def _find_grid_split_pixels(self, data):
        raise NotImplementedError()

    def matrix_layout(self):
        """Returns an array showing the layout of the matrices returned by `find_grid`."""
        n_groups = self.v_split * self.h_split
        return np.reshape(np.arange(n_groups), (self.v_split, self.h_split))


class InvalidDivision(Exception):
    """Raised when the data can't be divided into regions."""


class InvalidHorizontalDivision(InvalidDivision):
    """Raised when the data can't be divided into horizontal regions."""


class InvalidVerticalDivision(InvalidDivision):
    """Raised when the data can't be divided into vertical regions."""
