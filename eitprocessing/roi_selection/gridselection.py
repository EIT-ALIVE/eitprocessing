import bisect
import itertools
import warnings
from dataclasses import dataclass
from dataclasses import field
from enum import auto
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from strenum import LowercaseStrEnum
from . import ROISelection


class DivisionMethod(LowercaseStrEnum):
    geometrical = auto()
    geometrical_split_pixels = auto()
    physiological = auto()


# DivisionMethod.geometrical_split_pixels == "geometrical_split_pixels"

@dataclass
class GridSelection(ROISelection):
    """Create regions of interest by division into a grid.

    GridSelection allows for the creation of a list of 2D arrays that can be used to divide a two- or
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

    If the number of rows or columns can not be split evenly, a row or column can be split among two
    regions. This behaviour is controlled by `split_rows` and `split_columns`.

    If `split_rows` is `geometrical` (default), rows will not be split between two groups. A warning will
    be shown stating regions don't contain equal numbers of rows. The regions towards the top will
    be larger. E.g., when a (5, 2) array is split in two vertical regions, the first region will
    contain the first three rows, and the second region the last two rows.

    If `split_rows` is `geometrical_split_pixels`, e.g. a (5, 2) array that is split in two vertical regions, 
    the first region will contain the first two rows and half of each pixel of the third row. The second
    region contains half of each pixel in the third row, and the last two rows.

    If `split_rows` is `physiological`, an array is split so that the cumulative sum of the rows in each region
    is exactly equal. For instance, if an array is split in two regions, a pixel row can contribute for 20% to 
    the first region and for 80% to the second region.  

    `split_columns` has the same effect on columns as `split_rows` has on rows.

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
        split_rows: Allows rows to be split over two regions.
        split_columns: Allows columns to be split over two regions.

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
    split_rows: DivisionMethod = DivisionMethod.geometrical
    split_columns: DivisionMethod = DivisionMethod.geometrical
    ignore_nan_rows: bool = True
    ignore_nan_columns: bool = True

    def _check_attribute_type(self, name, type_):
        """Checks whether an attribute is an instance of the given type."""
        attr = getattr(self, name)
        if not isinstance(attr, type_):
            message = f"Invalid type for `{name}`."
            message += f"Should be {type_}, not {type(attr)}."
            raise TypeError(message)

    def __post_init__(self):
        self._check_attribute_type("v_split", int)
        self._check_attribute_type("h_split", int)

        if self.v_split < 1:
            raise InvalidVerticalDivision("`v_split` can't be smaller than 1.")

        if self.h_split < 1:
            raise InvalidHorizontalDivision("`h_split` can't be smaller than 1.")

        self._check_attribute_type("split_columns", DivisionMethod)
        self._check_attribute_type("split_rows", DivisionMethod)
        self._check_attribute_type("ignore_nan_columns", bool)
        self._check_attribute_type("ignore_nan_rows", bool)

    def find_grid(self, data: NDArray) -> list[NDArray]:
        """
        Create 2D arrays to split a grid into regions.

        Create 2D arrays to split the given data into regions. The number of 2D
        arrays will equal the number regions, which is the multiplicaiton of
        `v_split` and `h_split`.

        Args:
            data (NDArray): a 2D array containing any numeric or np.nan data.

        Returns:
            list[NDArray]: a list of `n` 2D arrays where `n` is `v_split *
                h_split`.
        """
        grouping_method = {
            DivisionMethod.geometrical: self._create_grouping_vector_no_split_pixels,
            DivisionMethod.geometrical_split_pixels: self._create_grouping_vector_split_pixels,
            DivisionMethod.physiological: self._create_grouping_vector_physiological
        }

        horizontal_grouping_vectors = grouping_method[self.split_columns](
            data, horizontal=True, n_groups=self.h_split
        )

        vertical_grouping_vectors = grouping_method[self.split_rows](
            data, horizontal=False, n_groups=self.v_split
        )

        matrices = []
        for vertical, horizontal in itertools.product(
            vertical_grouping_vectors, horizontal_grouping_vectors
        ):
            matrix = np.outer(vertical, horizontal)
            matrix[np.isnan(data)] = np.nan
            matrices.append(matrix)

        return matrices

    def _create_grouping_vector_no_split_pixels(  # pylint: disable=too-many-locals
        self,
        data: NDArray,
        horizontal: bool,
        n_groups: int,
    ) -> list[NDArray]:
        """Create a grouping vector to split vector into `n` groups not
        allowing split elements."""

        axis = 0 if horizontal else 1

        if (horizontal and self.ignore_nan_columns) or (
            not horizontal and self.ignore_nan_rows
        ):
            is_numeric = ~np.isnan(data)
            numeric_vector_indices = np.argwhere(is_numeric.sum(axis) > 0)
            first_numeric_vector = numeric_vector_indices.min()
            last_vector_numeric = numeric_vector_indices.max()
        else:
            first_numeric_vector = 0
            last_vector_numeric = data.shape[1 - axis] - 1

        n_vectors = last_vector_numeric - first_numeric_vector + 1

        if n_groups > n_vectors:
            if horizontal:  # pylint: disable=no-else-raise
                raise InvalidHorizontalDivision(
                    "The number horizontal regions is larger than the "
                    f"number of available columns ({n_vectors})."
                )
            else:
                raise InvalidVerticalDivision(
                    "The number vertical regions is larger than the "
                    f"number of available rows ({n_vectors})."
                )

        n_vectors_per_region = n_vectors / n_groups

        if n_vectors_per_region % 1 > 0:
            if horizontal:
                warnings.warn(
                    "The horizontal regions will not have an equal number of "
                    f"columns. {n_vectors} is not equally divisible by {n_groups}.",
                    UnevenHorizontalDivision,
                )
            else:
                warnings.warn(
                    "The vertical regions will not have an equal number of "
                    f"columns. {n_vectors} is not equally divisible by {n_groups}.",
                    UnevenVerticalDivision,
                )

        region_boundaries = [
            first_numeric_vector
            + bisect.bisect_left(np.arange(n_vectors) / n_vectors_per_region, c)
            for c in range(n_groups + 1)
        ]

        vectors = []
        for start, end in itertools.pairwise(region_boundaries):
            vector = np.ones(data.shape[1 - axis])
            vector[:start] = 0.0
            vector[end:] = 0.0
            vectors.append(vector)

        return vectors

    def _create_grouping_vector_split_pixels(  # pylint: disable=too-many-locals
        self,
        matrix: NDArray,
        horizontal: bool,
        n_groups: int,
    ) -> list[NDArray]:
        """Create a grouping vector to split vector into `n` groups allowing
        split elements."""

        axis = 0 if horizontal else 1

        # create a vector that is nan if the entire column/row is nan, 1 otherwise
        vector_is_nan = np.all(np.isnan(matrix), axis=axis)
        vector = np.ones(vector_is_nan.shape)

        if (horizontal and self.ignore_nan_columns) or (
            not horizontal and self.ignore_nan_rows
        ):
            vector[vector_is_nan] = np.nan

            # remove non-numeric (nan) elements at vector ends
            # nan elements between numeric elements are kept
            numeric_element_indices = np.argwhere(~np.isnan(vector))
            first_num_element = numeric_element_indices.min()
            last_num_element = numeric_element_indices.max()
        else:
            first_num_element = 0
            last_num_element = len(vector) - 1

        n_elements = last_num_element - first_num_element + 1

        group_size = n_elements / n_groups

        if group_size < 1:
            if horizontal:
                warnings.warn(
                    f"The number horizontal regions ({n_groups}) is larger than the "
                    f"number of available columns ({n_elements}).",
                    MoreHorizontalGroupsThanColumns,
                )
            else:
                warnings.warn(
                    f"The number vertical regions ({n_groups}) is larger than the "
                    f"number of available rows ({n_elements}).",
                    MoreVerticalGroupsThanRows,
                )

        # find the right boundaries (upper values) of each group
        right_boundaries = (np.arange(n_groups) + 1) * group_size
        right_boundaries = right_boundaries[:, np.newaxis]  # converts to row vector

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

        # convert to list of vectors
        final = [final[n, :] for n in range(final.shape[0])]

        return final

    def _create_grouping_vector_physiological(  # pylint: disable=too-many-locals
        self,
        matrix: NDArray,
        horizontal: bool,
        n_groups: int,
    ) -> list[NDArray]:

        """Create a grouping vector to split vector into `n` groups allowing
        split elements."""

        axis = 0 if horizontal else 1

        # create a vector that is nan if the entire column/row is nan, 1 otherwise
        vector_is_nan = np.all(np.isnan(matrix), axis=axis)
        vector = np.ones(vector_is_nan.shape)

        if (horizontal and self.ignore_nan_columns) or (
            not horizontal and self.ignore_nan_rows
        ):
            vector[vector_is_nan] = np.nan

            # remove non-numeric (nan) elements at vector ends
            # nan elements between numeric elements are kept
            numeric_element_indices = np.argwhere(~np.isnan(vector))
            first_num_element = numeric_element_indices.min()
            last_num_element = numeric_element_indices.max()
        else:
            first_num_element = 0
            last_num_element = len(vector) - 1

        n_elements = last_num_element - first_num_element + 1

        group_size = n_elements / n_groups

        if group_size < 1:
            if horizontal:
                warnings.warn(
                    f"The number horizontal regions ({n_groups}) is larger than the "
                    f"number of available columns ({n_elements}).",
                    MoreHorizontalGroupsThanColumns,
                )
            else:
                warnings.warn(
                    f"The number vertical regions ({n_groups}) is larger than the "
                    f"number of available rows ({n_elements}).",
                    MoreVerticalGroupsThanRows,
                )

        sum_along_axis = np.nansum(matrix, axis=axis)
        relative_sum_along_axis = sum_along_axis / np.nansum(matrix)
        relative_cumsum_along_axis = np.cumsum(relative_sum_along_axis)

        lower_bounds = np.arange(n_groups) / n_groups
        upper_bounds = (np.arange(n_groups) + 1) / n_groups

        # Otherwise the first row will not fall in the first region (because they are 0) 
        # and last rows will not fall in the last region, because they reach 1.0
        lower_bounds[0] = -np.inf
        upper_bounds[-1] = np.inf

        row_in_region = []

        for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
            row_in_region.append(
                np.logical_and(
                    relative_cumsum_along_axis > lower_bound, relative_cumsum_along_axis <= upper_bound
                )
            )

        row_in_region = np.array(row_in_region).T
        final = row_in_region.astype(float)

        # find initial region for each row
        initial_regions = np.apply_along_axis(np.flatnonzero, 1, row_in_region).flatten()

        # find transitions between regions
        region_borders = np.flatnonzero(np.diff(initial_regions))

        # finds overlap in transition region
        for previous_region, (ventral_row, upper_bound) in enumerate(
            zip(region_borders, upper_bounds)
        ):
            dorsal_row = ventral_row + 1
            next_region = previous_region + 1
            a, b = relative_cumsum_along_axis[ventral_row], relative_cumsum_along_axis[dorsal_row]
            diff = b - a
            to_a = upper_bound - a
            fraction_to_a = to_a / diff
            fraction_to_b = 1 - fraction_to_a

            final[dorsal_row, previous_region] = fraction_to_a
            final[dorsal_row, next_region] = fraction_to_b
        final = final.T
        final = final * vector
        # convert to list of vectors
        final = [final[n, :] for n in range(final.shape[0])]

        return final

    def matrix_layout(self) -> NDArray:
        """Returns a 2D array showing the layout of the matrices returned by
        `find_grid`."""
        n_regions = self.v_split * self.h_split
        return np.reshape(np.arange(n_regions), (self.v_split, self.h_split))


class InvalidDivision(Exception):
    """Raised when the data can't be divided into regions."""


class InvalidHorizontalDivision(InvalidDivision):
    """Raised when the data can't be divided into horizontal regions."""


class InvalidVerticalDivision(InvalidDivision):
    """Raised when the data can't be divided into vertical regions."""


class DivisionWarning(Warning):
    pass


class UnevenDivision(DivisionWarning):
    """Warning for when a grid selection results in groups of uneven size."""


class UnevenHorizontalDivision(UnevenDivision):
    """Warning for when a grid selection results in horizontal groups of uneven size."""


class UnevenVerticalDivision(UnevenDivision):
    """Warning for when a grid selection results in vertical groups of uneven size."""


class MoreGroupsThanVectors(DivisionWarning):
    """Warning for when the groups outnumber the available vectors."""


class MoreVerticalGroupsThanRows(MoreGroupsThanVectors):
    """Warning for when the vertical groups outnumber the available rows."""


class MoreHorizontalGroupsThanColumns(MoreGroupsThanVectors):
    """Warning for when the horizontal groups outnumber the available rows."""


@dataclass
class VentralAndDorsal(GridSelection):
    """Split data into a ventral and dorsal region of interest."""

    v_split: Literal[2] = field(default=2, init=False)
    h_split: Literal[1] = field(default=1, init=False)
    split_rows = True


@dataclass
class RightAndLeft(GridSelection):
    """Split data into a right and left region of interest."""

    v_split: Literal[1] = field(default=1, init=False)
    h_split: Literal[2] = field(default=2, init=False)
    split_columns = False


@dataclass
class FourLayers(GridSelection):
    """Split data vertically into four layer regions of interest."""

    v_split: Literal[4] = field(default=4, init=False)
    h_split: Literal[1] = field(default=1, init=False)
    split_rows = True


@dataclass
class Quadrants(GridSelection):
    """Split data into four quadrant regions of interest."""

    v_split: Literal[2] = field(default=2, init=False)
    h_split: Literal[2] = field(default=2, init=False)
    split_columns = False
    split_rows = True
