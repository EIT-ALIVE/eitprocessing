import bisect
import itertools
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import Literal
import numpy as np
from . import ROISelection


@dataclass
class GridSelection(ROISelection):
    """Create regions of interest by division into a grid.

    GridSelection allows for the creation a list of 2D arrays that can be used to divide a two- or
    higher-dimensional array into several regions structured in a grid. An instance of
    GridSelection contains information about how to subdivide an input matrix. Calling
    `find_grid(data)`, where data is a 2D array, results in a list of arrays with the same
    dimension as `data`, each representing a single region. Each resulting 2D array contains the
    value False or 0 for pixels that do not belong to the region, and the value True, 1 or any
    number between 0 and 1 for pixels that (partly) belong to the region.

    Rows and columns at the edges of `data` that only contain NaN (not a number) values are
    ignored. E.g. a (32, 32) array where the first and last two rows and first and last two columns
    only contain NaN are split as if it is a (28, 28) array. The resulting arrays have the shape
    (32, 32) with the same rows and columns only containing NaN values.

    If the number of rows or columns can not split evenly, a row or column can be split among two
    regions. This behaviour is controlled by `split_pixels`.

    If `split_pixels` is `True`, e.g. a (2, 5) array that is split in two horizontal regions, the
    first region will contain the first two columns, and half of the third column. The second
    region contains half of the third columns, and the last column.

    If `split_pixels` is `False` (default), rows and columns will not be split. A warning will be
    shown stating regions don't contain equal numbers of rows/columns. The regions towards the top
    and left will be larger. E.g., when a (2, 5) array is split in two horizontal regions, the
    first region will contain the first three columns, and the second region the last two columns.

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
        n_rows, n_columns = data.shape

        if self.split_pixels:
            # TODO: create warning if number of regions > number of columns/rows
            return self._find_grid_split_pixels(data)

        if self.h_split > n_columns:
            raise InvalidHorizontalDivision(
                f"`h_split` ({self.h_split}) is larger than the number of columns ({n_columns})."
            )

        if self.v_split > n_rows:
            raise InvalidVerticalDivision(
                f"`v_split` ({self.v_split}) is larger than the number or rows ({n_rows})."
            )

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
                    f"The {orientation} regions will not have an equal number of "
                    f"{'columns' if horizontal else 'rows'}. "
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
        def create_grouping_vector(matrix, axis, n_groups):
            """Create a grouping vector to split vector into `n` groups."""

            # create a vector that is nan if the entire column/row is nan, 1 otherwise
            nan = np.all(np.isnan(matrix), axis=axis)
            vector = np.ones(nan.shape)
            vector[nan] = np.nan

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

        horizontal_grouping_vectors = create_grouping_vector(data, 0, self.h_split)
        vertical_grouping_vectors = create_grouping_vector(data, 1, self.v_split)

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

    def matrix_layout(self):
        """Returns an array showing the layout of the matrices returned by `find_grid`."""
        n_regions = self.v_split * self.h_split
        return np.reshape(np.arange(n_regions), (self.v_split, self.h_split))


class InvalidDivision(Exception):
    """Raised when the data can't be divided into regions."""


class InvalidHorizontalDivision(InvalidDivision):
    """Raised when the data can't be divided into horizontal regions."""


class InvalidVerticalDivision(InvalidDivision):
    """Raised when the data can't be divided into vertical regions."""


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
