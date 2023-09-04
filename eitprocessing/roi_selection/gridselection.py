import bisect
import itertools
import numpy as np
from . import ROISelection


class GridSelection(ROISelection):
    def __init__(
        self,
        v_split: int,
        h_split: int,
        split_pixels: bool = False,
    ):
        self.h_split = h_split
        self.v_split = v_split
        self.split_pixels = split_pixels

    def find_grid(self, data) -> list:
        n_rows = data.shape[0]
        n_columns = data.shape[1]
        if self.h_split > n_columns:
            raise ValueError(
                "can't split a matrix into more horizontal regions than columns"
            )

        if self.v_split > n_rows:
            raise ValueError(
                "can't split a matrix into more vertical regions than rows"
            )

        if self.split_pixels:
            raise NotImplementedError()
            return self._find_grid_split_pixels(data)

        return self._find_grid_no_split_pixels(data)

    def _find_grid_no_split_pixels(self, data):
        # Detect upper, lower, left and right column of grid selection,
        # i.e, find the first row and column where the sum of pixels in that column is not NaN

        is_numeric = ~np.isnan(data)

        def get_region_boundaries(axis):
            n_regions = self.h_split if axis == 0 else self.v_split
            num_numeric_cells_in_vector = is_numeric.sum(axis)
            vectors_with_numbers = np.argwhere(num_numeric_cells_in_vector > 0)
            first_vector_with_number = vectors_with_numbers.min()
            last_vector_with_numer = vectors_with_numbers.max()

            n_vectors = last_vector_with_numer - first_vector_with_number + 1

            n_vectors_per_region = n_vectors / n_regions

            region_boundaries = [
                first_vector_with_number
                + bisect.bisect_left(np.arange(n_vectors), c * n_vectors_per_region)
                for c in range(0, n_regions)
            ] + [last_vector_with_numer + 1]
            return region_boundaries

        h_boundaries = get_region_boundaries(0)
        v_boundaries = get_region_boundaries(1)

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
        pass

    def matrix_layout(self):
        """Returns an array showing the layout of the matrices returned by `find_grid`."""
        n_groups = self.v_split * self.h_split
        return np.reshape(np.arange(n_groups), (self.v_split, self.h_split))
