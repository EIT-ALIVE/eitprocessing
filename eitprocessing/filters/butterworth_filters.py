from dataclasses import InitVar, dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import signal

from eitprocessing.filters import TimeDomainFilter


@dataclass(kw_only=True)
class ButterworthFilter(TimeDomainFilter):
    """Butterworth filter for filtering in the time domain.

    Generates a low-pass, high-pass, band-pass or band-stop digital Butterworth filter of order
    `order`. Filters are created using cascaded second-order sections representation, providing
    better stability compared to the traditionally used transfer function (numerator/denominator or
    b/a representation).

    The `apply_filter()` method applies the filter to the provided data using forward-backward
    filtering. This minimizes the phase shift, and effectively doubles the order of the filter.

    ``ButterworthFilter`` is a wrapper of the `scipy.butter()` and `scipy.filtfilt()` functions:
        - https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.signal.butter.html
        - https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.signal.filtfilt.html

    Args:
        filter_type: The type of filter to create: a low pass, high pass, band pass or band stop
            filter.
        cutoff_frequency: Cutoff frequency or frequencies (in Hz). For low pass or high pass
            filters, `cutoff_frequency` is a scalar. For band pass or band stop filters,
            `cutoff_frequency` is a sequence containing two frequencies.
        order: Order of the filter. The effective order size is twice the given order, due to
            forward-backward filtering. Higher orders improve the effectiveness of a filter, but
            can result in unstable or incorrect filtering.
        sample_frequency: The sample frequency of the data to be filtered (in Hz).
        ignore_max_order: Whether to raise an exception if the order is larger than the maximum of
            10. Defaults to False.

    Examples:
        >>> t = np.arange(0, 100, 0.1)
        >>> signal = np.sin(t) + 0.1 * np.sin(10 * t)
        >>> lowpass_filter = ButterworthFilter(
        ...     filter_type='lowpass',
        ...     cutoff_frequenct=45,
        ...     order=4,
        ...     sample_frequency=250
        ... )
        >>> filtered_signal = lowpass_filter.apply_filter(signal)
    """

    filter_type: Literal["lowpass", "highpass", "bandpass", "bandstop"]
    cutoff_frequency: float | tuple[float]
    order: int
    sample_frequency: float
    ignore_max_order: InitVar[bool] = False

    def __post_init__(self, ignore_max_order: bool):
        self._check_init(ignore_max_order)
        self._set_filter_type_class()

    def _set_filter_type_class(self) -> None:
        if (
            isinstance(self, ButterworthFilter)
            and self.__class__ != ButterworthFilter
            and self.__class__.filter_type != self.filter_type
        ):
            msg = f"conflicting type info; `filter_type={self.filter_type}` does not match {self.__class__}."
            raise TypeError(msg)

        # Note that this way of re-assigning classes is considered to be a bad practice
        # (https://tinyurl.com/2x2cea6h), but the objections raised don't seem to be prohibtive.
        cls = FILTER_TYPES[self.filter_type]
        self.__class__ = cls

    def _check_init(self, ignore_max_order: bool) -> None:  # noqa:C901
        """Check the arguments of __init__ and raise exceptions when they don't meet requirements.

        Raises:
            ValueError: if the `filter_type` is unknown.
            TypeError: if the cutoff frequency isn't numeric (low/high pass filters) or a tuple
                (band pass/stop filters).
            ValueError: if the number of provided cutoff frequencies is not 2 (band pass/stop
                filters).
            TypeError: if the tuple `cutoff_frequency` does not contains numeric values (band
                pass/stop filters).
            ValueError: if the order is lower than `MIN_ORDER` or higher than `MAX_ORDER`. Can be
                prevented when the order is higher than `MAX_ORDER` with `ignore_max_order = True`.
            TypeError: if the sample frequency is not numeric.
            ValueError: if the sample frequency is 0 or negative.
        """
        if self.filter_type in ("lowpass", "highpass"):
            if not isinstance(self.cutoff_frequency, int | float):
                msg = f"Invalid `cutoff_frequency`. Must be an integer or float, not {type(self.cutoff_frequency)}."
                raise TypeError(msg)

        elif self.filter_type in ("bandpass", "bandstop"):
            if isinstance(self.cutoff_frequency, list):
                self.cutoff_frequency = tuple(self.cutoff_frequency)
            elif not isinstance(self.cutoff_frequency, tuple):
                msg = f"Invalid `cutoff_frequency`. Must be a tuple, not {type(self.cutoff_frequency)}."
                raise TypeError(msg)

            if len(self.cutoff_frequency) != 2:  # noqa: PLR2004
                msg = f"Invalid `cutoff_frequency` ({self.cutoff_frequency}). Must have length 2."
                raise ValueError(msg)

            if not all(isinstance(value, int | float) for value in self.cutoff_frequency):
                msg = f"Invalid `cutoff_frequency` ({self.cutoff_frequency}). Must be a tuple containing two numbers."
                raise TypeError(msg)
        else:
            msg = f"Invalid `filter_type` ({self.filter_type}). Must be one of {', '.join(FILTER_TYPES.keys())}."
            raise ValueError(msg)

        if not isinstance(self.order, int):
            msg = f"Invalid `order`. Must be an int, not {type(self.order)}."
            raise TypeError(msg)

        if self.order < 1 or (self.order > MAX_ORDER and ignore_max_order is False):
            msg = (
                f"Invalid `order` ({self.order}). Must be between 1 and {MAX_ORDER}. "
                "To use higher values, set `ignore_max_order` to `True`."
            )
            raise ValueError(msg)

        if not isinstance(self.sample_frequency, int | float):
            msg = f"Invalid `sample_frequency` ({self.sample_frequency}). Must be a number."
            raise TypeError(msg)
        if self.sample_frequency <= 0:
            msg = f"Invalid `sample_frequency` ({self.sample_frequency}). Must be positive"
            raise ValueError(msg)

    def apply_filter(self, input_data: npt.ArrayLike, axis: int = -1) -> np.ndarray:
        """Apply the filter to the input data.

        Args:
            input_data: Data to be filtered. If the input data has more than one axis,
                the filter is applied to the last axis.
            axis: Data axis the filter should be applied to. This defaults to the last axis,
                assuming this to be the time axis of the input data.

        Returns:
            The filtered output with the same shape as the input data.
        """
        sos = signal.butter(
            N=self.order,
            Wn=self.cutoff_frequency,
            btype=self.filter_type,
            fs=self.sample_frequency,
            analog=False,
            output="sos",
        )

        return signal.sosfiltfilt(sos, input_data, axis=axis)


@dataclass(kw_only=True)
class LowPassFilter(ButterworthFilter):
    """Low-pass Butterworth filter for filtering in the time domain.

    ``LowPassFilter`` is a convenience class similar to ``ButterworthFilter``, where the
    `filter_type` is set to "lowpass".
    """

    available_in_gui = True
    filter_type: Literal["lowpass"] = "lowpass"


@dataclass(kw_only=True)
class HighPassFilter(ButterworthFilter):
    """High-pass Butterworth filter for filtering in the time domain.

    ``HighPassFilter`` is a convenience class similar to ``ButterworthFilter``, where the
    `filter_type` is set to "highpass".
    """

    available_in_gui = True
    filter_type: Literal["highpass"] = "highpass"


@dataclass(kw_only=True)
class BandStopFilter(ButterworthFilter):
    """Band-stop Butterworth filter for filtering in the time domain.

    ``BandStopFilter`` is a convenience class similar to ``ButterworthFilter``, where the
    `filter_type` is set to "bandstop".
    """

    available_in_gui = True
    filter_type: Literal["bandstop"] = "bandstop"


@dataclass(kw_only=True)
class BandPassFilter(ButterworthFilter):
    """Band-pass Butterworth filter for filtering in the time domain.

    ``BandPassFilter`` is a convenience class similar to ``ButterworthFilter``, where the
    `filter_type` is set to "bandpass".
    """

    available_in_gui = True
    filter_type: Literal["bandpass"] = "bandpass"


MAX_ORDER = 10
FILTER_TYPES = {
    "lowpass": LowPassFilter,
    "highpass": HighPassFilter,
    "bandpass": BandPassFilter,
    "bandstop": BandStopFilter,
}
