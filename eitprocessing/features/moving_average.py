from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from eitprocessing.sequence import Sequence


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
        window_length: the size of the window. Is enlarged by 1 if uneven.
        window_fun: window function, e.g. np.window.bartlett.
        padding_type: type of padding to apply. See np.pad().

    Returns:
        Moving average as a 1D array with the same length as `data`.
    """

    window_length: int
    window_fun: Callable | None = None
    padding_type: str = "edge"

    def apply(self, sequence: Sequence, data_type: str, label: str) -> np.ndarray:
        """Apply the moving average on the data.

        Args:
            sequence: the Sequence containing the data.
            data_type: should be "continuous".
            label: label of the continuous data to apply the moving average to.
        """
        if data_type != "continuous":
            msg = f"BreathDetection only works on continuous data, not {data_type}"
            raise NotImplementedError(msg)

        continuous_data = sequence.continuous_data[label]
        data: np.ndarray = continuous_data.values  # noqa: PD011
        sample_frequency = continuous_data.sample_frequency

        window_size = sample_frequency * self.window_length

        if window_size % 2 == 0:
            window_size += 1

        if window_size > len(data):
            window_size = int((len(data) - 1) / 2) + 1

        if self.window_fun:
            window = np.array(self.window_fun(window_size))
            window = window / np.sum(window)  # normalizes to an area of 1
        else:
            window = np.ones(window_size) / window_size

        padding_length = (window_size - 1) // 2
        padded_data = np.pad(data, padding_length, mode=self.padding_type)
        return np.convolve(padded_data, window, mode="valid")
