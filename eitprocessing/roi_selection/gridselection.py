import bisect
import itertools
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from . import ROISelection


@dataclass
class GridSelection(ROISelection):
    """Create regions of interest by division into a grid.

    GridSelection allows for the creation a list of 2D arrays that can be used to divide a two- or
    higher-dimensional array into several regions structured in a grid. An instance of
    GridSelection contains information about how to subdivide an input matrix. Calling
    `find_grid(data)`, where data is a 2D array, results in a list of arrays with the same
    dimension as `data`, each representing a single region. Each resulting 2D array contains the
    value 0 for pixels that do not belong to the region, and the value 1 or any number between 0
    and 1 for pixels that (partly) belong to the region.

    Rows and columns at the edges of `data` that only contain NaN (not a number) values are
    ignored. E.g. a (32, 32) array where the first and last two rows and first and last two columns
    only contain NaN are split as if it is a (28, 28) array. The resulting arrays have the shape
    (32, 32) with the same cells as the input data containing NaN values.

    If the number of rows or columns can not split evenly, a row or column can be split among two
    regions. This behaviour is controlled by `split_pixels`.

    If `split_pixels` is `False` (default), rows and columns will not be split. A warning will be
    shown stating regions don't contain equal numbers of rows/columns. The regions towards the top
    and left will be larger. E.g., when a (2, 5) array is split in two horizontal regions, the
    first region will contain the first three columns, and the second region the last two columns.

    If `split_pixels` is `True`, e.g. a (2, 5) array that is split in two horizontal regions, the
    first region will contain the first two columns, and half of the third column. The second
    region contains half of the third columns, and the last column.

    Regions are ordered according to C indexing order. The `matrix_layout()` method provides a map
    showing how the regions are ordered.

    Common grids are pre-defined:
    - VentralAndDorsal: vertically divided into ventral and dorsal;
    - RightAndLeft: horizontally divided into anatomical right and left; NB: anatomical right is
    the left side of the matrix;
    - FourLayers: vertically divided into ventral, mid-ventral, mid-dorsal and dorsal;
    - Quadrants: vertically and horizontally divided into four quadrants.

    Args:
        v_split: The number of vertical regions. Must be 1 or larger.
        h_split: The number of horizontal regions. Must be 1 or larger.
        split_pixels: Allows rows and columns to be split over two regions.

    Examples:
        >>> pixel_map = array([[ 1,  2,  3],
                               [ 4,  5,  6],
                               [ 7,  8,  9],
                               [10, 11, 12],
                               [13, 14, 15],
                               [16, 17, 18]])
        >>> gs = GridSelection(3, 1, split_pixels=False)
        >>> matrices = gs.find_grid(pixel_map)
        >>> matrices[0] * pixel_map
        array([[1, 2, 3],
               [4, 5, 6],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0]])
        >>> gs.matrix_layout()
        array([[0],
               [1],
               [2]])
        >>> gs2 = GridSelection(2, 2, split_pixels=True)
        >>> matrices2 = gs.find_grid(pixel_map)
        >>> gs2.matrix_layout()
        array([[0, 1],
               [2, 3]])
        >>> matrices2[2]
        array([[0. , 0. , 0. ],
               [0. , 0. , 0. ],
               [0. , 0. , 0. ],
               [1. , 0.5, 0. ],
               [1. , 0.5, 0. ],
               [1. , 0.5, 0. ]])
    """

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

    def find_grid(self, data) -> list:
        function = (
            self._create_grouping_vector_split_pixels
            if self.split_pixels
            else self._create_grouping_vector_no_split_pixels
        )
        horizontal_grouping_vectors = function(data, "horizontal", self.h_split)

        function = (
            self._create_grouping_vector_split_pixels
            if self.split_pixels
            else self._create_grouping_vector_no_split_pixels
        )
        vertical_grouping_vectors = function(data, "vertical", self.v_split)

        matrices = []
        for vertical, horizontal in itertools.product(
            vertical_grouping_vectors, horizontal_grouping_vectors
        ):
            matrix = np.ones(data.shape)
            matrix[np.isnan(data)] = np.nan
            matrix *= horizontal
            matrix *= vertical[:, None]  # [:, None] converts to a column vector
            matrices.append(matrix)

        return matrices

    @staticmethod
    def _create_grouping_vector_no_split_pixels(  # pylint: disable=too-many-locals
        data: NDArray, orientation: Literal["horizontal", "vertical"], n_regions: int
    ) -> list[NDArray]:
        is_numeric = ~np.isnan(data)
        axis = 0 if orientation == "horizontal" else 1
        numeric_vector_indices = np.argwhere(is_numeric.sum(axis) > 0)
        first_numeric_vector = numeric_vector_indices.min()
        last_vector_numeric = numeric_vector_indices.max()

        n_vectors = last_vector_numeric - first_numeric_vector + 1

        if n_regions > n_vectors:
            if orientation == "horizontal":  # pylint: disable=no-else-raise
                raise InvalidHorizontalDivision(
                    f"The number horizontal regions is larger than the "
                    f"number of available columns ({n_vectors})."
                )
            else:
                raise InvalidVerticalDivision(
                    f"The number vertical regions is larger than the "
                    f"number of available rows ({n_vectors})."
                )

        n_vectors_per_region = n_vectors / n_regions

        if n_vectors_per_region % 1 > 0:
            if orientation == "horizontal":
                warnings.warn(
                    f"The horizontal regions will not have an equal number of "
                    f"columns. {n_vectors} is not equally divisible by {n_regions}.",
                    UnevenHorizontalDivision,
                )
            else:
                warnings.warn(
                    f"The vertical regions will not have an equal number of "
                    f"columns. {n_vectors} is not equally divisible by {n_regions}.",
                    UnevenVerticalDivision,
                )

        region_boundaries = [
            first_numeric_vector
            + bisect.bisect_left(np.arange(n_vectors) / n_vectors_per_region, c)
            for c in range(n_regions + 1)
        ]

        vectors = []
        for start, end in itertools.pairwise(region_boundaries):
            vector = np.ones(data.shape[1 - axis])
            vector[:start] = 0
            vector[end:] = 0
            vectors.append(vector)

        return vectors

    @staticmethod
    def _create_grouping_vector_split_pixels(  # pylint: disable=too-many-locals
        matrix: NDArray, orientation: Literal["horizontal", "vertical"], n_groups: int
    ) -> NDArray:
        """Create a grouping vector to split vector into `n` groups."""
        axis = 0 if orientation == "horizontal" else 1

        # create a vector that is nan if the entire column/row is nan, 1 otherwise
        vector_is_nan = np.all(np.isnan(matrix), axis=axis)
        vector = np.ones(vector_is_nan.shape)
        vector[vector_is_nan] = np.nan

        # remove non-numeric (nan) elements at vector ends
        # nan elements between numeric elements are kept
        numeric_element_indices = np.argwhere(~np.isnan(vector))
        first_num_element = numeric_element_indices.min()
        last_num_element = numeric_element_indices.max()
        n_elements = last_num_element - first_num_element + 1

        group_size = n_elements / n_groups

        # find the right boundaries (upper values) of each group
        right_boundaries = (np.arange(n_groups) + 1) * group_size
        right_boundaries = right_boundaries[:, None]  # converts it to a row vector

        # each row in the base represents one group
        base = np.tile(np.arange(n_elements), (n_groups, 1))

        # if the element number is higher than the split, it does not belong in this group
        element_contribution_to_group = right_boundaries - base
        element_contribution_to_group[element_contribution_to_group < 0] = 0

        # if the element to the right is a full group size, this element is ruled out
        rule_out = element_contribution_to_group[:, 1:] >= group_size
        element_contribution_to_group[:, :-1][rule_out] = 0

        # elements have a maximum value of 1
        element_contribution_to_group = np.fmin(element_contribution_to_group, 1)

        # if this element is already represented in the previous group (row), subtract that
        element_contribution_to_group[1:] -= element_contribution_to_group[:-1]
        element_contribution_to_group[element_contribution_to_group < 0] = 0

        # element_contribution_to_group only represents non-nan elements
        # insert into final including non-nan elements
        final = np.full((n_groups, len(vector)), np.nan)
        final[
            :, first_num_element : last_num_element + 1
        ] = element_contribution_to_group
        return final

    def matrix_layout(self) -> NDArray:
        """Returns an array showing the layout of the matrices returned by `find_grid`."""
        n_regions = self.v_split * self.h_split
        return np.reshape(np.arange(n_regions), (self.v_split, self.h_split))


class InvalidDivision(Exception):
    """Raised when the data can't be divided into regions."""


class InvalidHorizontalDivision(InvalidDivision):
    """Raised when the data can't be divided into horizontal regions."""


class InvalidVerticalDivision(InvalidDivision):
    """Raised when the data can't be divided into vertical regions."""


class UnevenDivision(Warning):
    """Warning for when a grid selection results in groups of uneven size."""


class UnevenHorizontalDivision(UnevenDivision):
    """Warning for when a grid selection results in horizontal groups of uneven size."""


class UnevenVerticalDivision(UnevenDivision):
    """Warning for when a grid selection results in vertical groups of uneven size."""


@dataclass
class VentralAndDorsal(GridSelection):
    """Split data into a ventral and dorsal region of interest."""

    v_split: Literal[2] = field(default=2, init=False)
    h_split: Literal[1] = field(default=1, init=False)
    split_pixels: bool = False


@dataclass
class RightAndLeft(GridSelection):
    """Split data into a right and left region of interest."""

    v_split: Literal[1] = field(default=1, init=False)
    h_split: Literal[2] = field(default=2, init=False)
    split_pixels: bool = False


@dataclass
class FourLayers(GridSelection):
    """Split data vertically into four layer regions of interest."""

    v_split: Literal[4] = field(default=4, init=False)
    h_split: Literal[1] = field(default=1, init=False)
    split_pixels: bool = False


@dataclass
class Quadrants(GridSelection):
    """Split data into four quadrant regions of interest."""

    v_split: Literal[2] = field(default=2, init=False)
    h_split: Literal[2] = field(default=2, init=False)
    split_pixels: bool = False
