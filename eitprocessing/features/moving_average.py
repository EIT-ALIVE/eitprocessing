from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class MovingAverage:
    """Calculate the moving average of the data.

    The moving average is calculated using a convolution with a window. The
    window length (in seconds) is determined by the attribute
    `averaging_window_duration`. The shape of the window is determined by
    `averaging_window_fun`, which should be a callable that takes an
    integer `M` and returns an array-like sequence containing a window with
    length `M` and area 1.

    Before convolution the data is padded. The padding type is 'edge' by
    default. See `np.pad()` for more information. Padding adds values at the
    start and end with the first/last value, to more accurately determine the
    average at the boundaries of the data.

    Args:
      window_size: the size of the window. Is enlarged by 1 if uneven.
      window_fun: window function, e.g. np.window.bartlett.

    Returns:
        Moving average as a 1D array with the same length as `data`.
    """

    window_size: int
    window_fun: Callable | None = None
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

        if self.window_fun:
            window = np.array(self.window_fun(window_size))
            window = window / np.sum(window)  # normalizes to an area of 1
        else:
            window = np.ones(window_size) / window_size

        padding_length = (window_size - 1) // 2
        padded_data = np.pad(data, padding_length, mode=self.padding_type)
        return np.convolve(padded_data, window, mode="valid")
