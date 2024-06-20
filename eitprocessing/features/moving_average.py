from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class MovingAverage:
    """Algorithm for calculating the moving average of the data.

    This class provides a method for calculating of the moving average of a 1D signal by convolution with a window with
    a given size. If not window function is provided, all samples within that window contribute equally to the moving
    average. If a window function is provided, the samples are weighed according to the values in the window function.

    Before convolution the data is padded. The padding type is 'edge' by default. See `np.pad()` for more information.
    Padding adds values at the start and end with the first/last value, to more accurately determine the average at the
    boundaries of the data.

    Args:
        window_size: the number of data points in the averaging window. Should be odd; is increased by 1 if even.
        window_function: window function, e.g. np.bartlett.
        padding_type: see `np.pad()`.

    Returns:
        np.ndarray: moving average as a 1D array with the same length as `data`.
    """

    window_size: int
    window_function: Callable | None = None
    padding_type: str = "edge"

    def __post_init__(self):
        if self.window_size % 2 == 0:
            self.window_size += 1

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the moving average on the data.

        Args:
            data (NDArray): input data as 1D array
        """
        window_size = self.window_size
        if window_size > len(data):
            window_size = int((len(data) - 1) / 2) * 2 + 1

        if self.window_function:
            window = np.array(self.window_function(window_size))
            window = window / np.sum(window)  # normalizes to an area of 1
        else:
            window = np.ones(window_size) / window_size

        padding_length = (window_size - 1) // 2
        padded_data = np.pad(data, padding_length, mode=self.padding_type)
        return np.convolve(padded_data, window, mode="valid")
