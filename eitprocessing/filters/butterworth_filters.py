from typing import Literal
import numpy as np
import numpy.typing as npt
from scipy import signal
from . import TimeDomainFilter


MIN_ORDER = 1
MAX_ORDER = 10
FILTER_TYPES = ["lowpass", "highpass", "bandpass", "bandstop"]


class ButterworthFilter(TimeDomainFilter):
    """Butterworth filter for filtering in the time domain.

    Generates a low-pass, high-pass, band-pass or band-stop digital Butterworth filter of order
    `order`.

    ``ButterworthFilter`` is a wrapper of the `scipy.butter()` and `scipy.filtfilt()` functions.

    Args:
        filter_type: The type of filter to create.
        cutoff_frequency: Single frequency (lowpass or highpass filter) or tuple containing two
            frequencies (bandpass and bandstop filters).
        order: Filter order.
        sample_frequency: Sample frequency of the data to be filtered.
        override_order: Whether to raise an exception if the order is larger than the maximum of
            10. Defaults to False.

    Examples:
        >>> t = np.arange(0, 100, 0.1)
        >>> signal = np.sin(t) + 0.1 * np.sin(10 * t)
        >>> lowpass_filter = ButterworthFilter('lowpass', 45, 4, 250)
        >>> filtered_signal = lowpass_filter.apply_filter(signal)
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"],
        cutoff_frequency: float | tuple[float, float],
        order: int,
        sample_frequency: float,
        override_order: bool = False,
    ):
        if filter_type not in FILTER_TYPES:
            raise ValueError(
                "The filter type should be one of "
                f"{', '.join(FILTER_TYPES)}, not '{filter_type}'."
            )
        self.filter_type = filter_type

        if filter_type in ("lowpass", "highpass"):
            if not isinstance(cutoff_frequency, (int, float)):
                raise TypeError("`cutoff_frequency` should be an integer or float")

        elif filter_type in ("bandpass", "bandstop"):
            if not isinstance(cutoff_frequency, tuple):
                raise TypeError("`cutoff_frequency` should be a tuple")

            if len(cutoff_frequency) != 2:
                raise ValueError("`cutoff_frequency` should have length 2")

            if not all(isinstance(value, (int, float)) for value in cutoff_frequency):
                raise TypeError(
                    "`cutoff_frequency` should be a tuple containing two numbers"
                )

        self.cutoff_frequency = cutoff_frequency

        if order < MIN_ORDER or (order > MAX_ORDER and override_order is False):
            raise ValueError(
                f"Order should be between {MIN_ORDER} and {MAX_ORDER}. "
                "To use higher values, set `override_order` to `True`."
            )
        self.order = order

        if not isinstance(sample_frequency, (int, float)):
            raise TypeError("`sample_frequency` should be a number")
        if sample_frequency <= 0:
            raise ValueError("`sample_frequency` should be positive")
        self.sample_frequency = sample_frequency

    def __eq__(self, other: "ButterworthFilter"):
        """Return True if other is a ``ButterworthFilter``, and attributes match."""
        if not isinstance(other, ButterworthFilter):
            return False
        return self.__dict__ == other.__dict__

    def apply_filter(self, input_data: npt.ArrayLike) -> np.ndarray:
        """Apply the filter to the input data.

        Args:
            input_data: Data to be filtered. If the input data has more than one axis,
                the filter is applied to the last axis.

        Returns:
            The filtered output with the same shape as the input data.
        """
        b, a = signal.butter(
            N=self.order,
            Wn=self.cutoff_frequency,
            btype=self.filter_type,
            fs=self.sample_frequency,
            analog=False,
            output="ba",
        )

        filtered_data = signal.filtfilt(b, a, input_data, axis=-1)
        return filtered_data


class SpecifiedButterworthFilter(ButterworthFilter):
    """Superclass of specified convenience classes based on ``ButterworthFilter``."""

    available_in_gui = False

    def __init__(
        self,
        cutoff_frequency: float | tuple[float, float],
        order: int,
        sample_frequency: float,
        override_order: bool = False,
        **kwargs,
    ):
        if "filter_type" in kwargs:
            filter_type = kwargs.pop("filter_type")
            if filter_type != self.filter_type:
                raise AttributeError("`filter_type` should not be supplied.")

        if len(kwargs):
            raise AttributeError(
                f"got an unexpect keyword argument: {', '.join(kwargs.keys())}"
            )

        ButterworthFilter.__init__(
            self,
            filter_type=self.filter_type,
            cutoff_frequency=cutoff_frequency,
            order=order,
            sample_frequency=sample_frequency,
            override_order=override_order,
            **kwargs,
        )


class LowPassFilter(SpecifiedButterworthFilter):
    """Low-pass Butterworth filter for filtering in the time domain.

    ``LowPassFilter`` is a convenience class similar to ``ButterworthFilter``, where the
    `filter_type` is set to "lowpass".
    """

    available_in_gui = True
    filter_type = "lowpass"


class HighPassFilter(SpecifiedButterworthFilter):
    """High-pass Butterworth filter for filtering in the time domain.

    ``HighPassFilter`` is a convenience class similar to ``ButterworthFilter``, where the
    `filter_type` is set to "highpass".
    """

    available_in_gui = True
    filter_type = "highpass"


class BandStopFilter(SpecifiedButterworthFilter):
    """Band-stop Butterworth filter for filtering in the time domain.

    ``BandStopFilter`` is a convenience class similar to ``ButterworthFilter``, where the
    `filter_type` is set to "bandstop".
    """

    available_in_gui = True
    filter_type = "bandstop"


class BandPassFilter(SpecifiedButterworthFilter):
    """Band-pass Butterworth filter for filtering in the time domain.

    ``BandPassFilter`` is a convenience class similar to ``ButterworthFilter``, where the
    `filter_type` is set to "bandpass".
    """

    available_in_gui = True
    filter_type = "bandpass"
