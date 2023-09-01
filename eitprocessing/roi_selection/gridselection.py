import numpy as np
from . import ROISelection


class GridSelection(ROISelection):
    def __init__(
        self,
        rows: int,
        cols: int,
        split_pixels: bool = False,
    ):
        self.rows = rows
        self.cols = cols
        self.split_pixels = split_pixels

    def find_grid(self, data):
        # Detect upper, lower, left and right column of grid selection, 
        # i.e, find the first row and column where the sum of pixels in that column is not NaN
        numeric_cols = (~np.isnan(data)).sum(0)
        cols_with_numbers = np.argwhere(numeric_cols > 0)
        first_col_with_number = cols_with_numbers.min()
        last_col_with_numer = cols_with_numbers.max()

        n_columns = last_col_with_numer - first_col_with_number + 1

        n_columns_per_group = n_columns / self.cols
        
        splits = np.arange(self.cols) * n_columns_per_group + first_col_with_number


        