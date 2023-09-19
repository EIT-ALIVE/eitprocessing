from collections.abc import Sequence
from dataclasses import InitVar
from dataclasses import dataclass
from typing import Literal
import numpy as np
import numpy.typing as npt
from scipy import signal
from . import TimeDomainFilter


@dataclass(kw_only=True)
class ButterworthFilter(TimeDomainFilter):
    """Butterworth filter for filtering in the time domain.

    Generates a low-pass, high-pass, band-pass or band-stop digital Butterworth filter of order
    `order`.

    ``ButterworthFilter`` is a wrapper of the `scipy.butter()` and `scipy.filtfilt()` functions:
    - https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.signal.butter.html
    - https://docs.scipy.org/doc/scipy-1.10.1/reference/generated/scipy.signal.filtfilt.html

    Args:
        filter_type: The type of filter to create: a low pass, high pass, band pass or band stop
            filter.
        cutoff_frequency: Cutoff frequency or frequencies (in Hz). For low pass or high pass
            filters, `cutoff_frequency` is a scalar. For band pass or band stop filters,
            `cutoff_frequency` is a sequence containing two frequencies.
        order: Order of the filter. High-order filters can result in instable or incorrect
            filtering.
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
    cutoff_frequency: float | Sequence[float]
    order: int
    sample_frequency: float
    ignore_max_order: InitVar[bool] = False

    def __post_init__(self, ignore_max_order):
        self._check_init(ignore_max_order)
        self._set_filter_type_class()

    def _set_filter_type_class(self):
        self._check_filter_type(self.filter_type)
        if (
            isinstance(self, ButterworthFilter)
            and self.__class__ != ButterworthFilter
            and self.__class__.filter_type != self.filter_type
        ):
            raise TypeError(
                f"conflicting type info; `filter_type={self.filter_type}` does not match {self.__class__}."
            )
        
        # Note that this way of re-assigning classes is considered to be a bad practice
        # (https://tinyurl.com/2x2cea6h), but the objections raised don't seem to be prohibtive.
        cls = FILTER_TYPES[self.filter_type]
        self.__class__ = cls

    @staticmethod
    def _check_filter_type(filter_type):
        if filter_type not in FILTER_TYPES:
            raise ValueError(
                "The filter type should be one of "
                f"{', '.join(FILTER_TYPES.keys())}, not '{filter_type}'."
            )

    def _check_init(self, ignore_max_order):
        """
        Check the arguments of __init__ and raise exceptions when they don't meet requirements.

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
        self._check_filter_type(self.filter_type)

        if self.filter_type in ("lowpass", "highpass"):
            if not isinstance(self.cutoff_frequency, (int, float)):
                raise TypeError("`cutoff_frequency` should be an integer or float")

        else:  # implies self.filter_type in ("bandpass", "bandstop"):
            if not isinstance(self.cutoff_frequency, tuple):
                if isinstance(self.cutoff_frequency, Sequence) and not isinstance(
                    self.cutoff_frequency, (str, bytes), 
                ):
                    try:
                        self.cutoff_frequency = tuple(self.cutoff_frequency)
                    except Exception as e:
                        raise TypeError(
                            f"can't convert sequence {self.cutoff_frequency} to tuple"
                        ) from e
                else:
                    raise TypeError(
                        "`cutoff_frequency` should be a sequence of 2 numbers"
                    )

            if len(self.cutoff_frequency) != 2:
                raise ValueError("`cutoff_frequency` should have length 2")

            if not all(
                isinstance(value, (int, float)) for value in self.cutoff_frequency
            ):
                raise TypeError("`cutoff_frequency` be a sequence of two numbers")

        if self.order < MIN_ORDER or (
            self.order > MAX_ORDER and ignore_max_order is False
        ):
            raise ValueError(
                f"Order should be between {MIN_ORDER} and {MAX_ORDER}. "
                "To use higher values, set `ignore_max_order` to `True`."
            )

        if not isinstance(self.sample_frequency, (int, float)):
            raise TypeError("`sample_frequency` should be a number")
        if self.sample_frequency <= 0:
            raise ValueError("`sample_frequency` should be positive")

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

        filtered_data = signal.sosfiltfilt(sos, input_data, axis=axis)
        return filtered_data


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


MIN_ORDER = 1
MAX_ORDER = 10
FILTER_TYPES = {
    "lowpass": LowPassFilter,
    "highpass": HighPassFilter,
    "bandpass": BandPassFilter,
    "bandstop": BandStopFilter,
}
