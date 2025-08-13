import copy
import math
import warnings
from dataclasses import dataclass
from typing import TypeVar, cast, overload

import numpy as np
from scipy import signal

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.filters import TimeDomainFilter
from eitprocessing.plotting.filter import FilterPlotting
from eitprocessing.utils import _CaptureFunc, make_capture

MINUTE = 60
NOISE_FREQUENCY_LIMIT: float = 220 / MINUTE
DEFAULT_AXIS: int = 0

# TODO: centralize settings (these should be shared with e.g. RateDetection)
UPPER_RESPIRATORY_RATE_LIMIT: float = 85 / MINUTE
UPPER_HEART_RATE_LIMIT: float = 210 / MINUTE

T = TypeVar("T", bound=np.ndarray | ContinuousData | EITData)


MISSING = object()


@dataclass(frozen=True, kw_only=True)
class MDNFilter(TimeDomainFilter):
    """Multiple Digital Notch filter.

    This filter is used to remove heart rate noise from data. A band stop filter removes heart rate
    ± the notch distance. This is repeated for every harmonic of the heart rate below the noise
    frequency limit. Lastly, a low pass filter removes noise above the noise frequency limit.

    By default, the notch distance is set to 0.166... Hz (10 BPM), and the noise frequency limit is
    set to 3.66... Hz (220 BPM).

    Warning:
        The respiratory and heart rate should be in provided Hz, not BPM. We recommend defining `MINUTE = 60` and using,
        e.g., `heart_rate=80 / MINUTE` to manually set the heart rate to 80 BPM.

    Args:
      respiratory_rate: the respiratory rate of the subject in Hz
      heart_rate: the heart rate of the subject in Hz
      noise_frequency_limit: the highest frequency to filter in Hz
      notch_distance: the half width of the band stop filter's frequency range
    """

    respiratory_rate: float
    heart_rate: float
    noise_frequency_limit: float = 220 / MINUTE
    notch_distance: float = 10 / MINUTE
    order: int = 10

    def __post_init__(self):
        if self.respiratory_rate > UPPER_RESPIRATORY_RATE_LIMIT:
            msg = (
                f"The provided respiratory rate ({self.respiratory_rate:.1f}) "
                f"is higher than {UPPER_RESPIRATORY_RATE_LIMIT} Hz "
                f"({UPPER_RESPIRATORY_RATE_LIMIT * MINUTE} BPM). "
                "Make sure to use the correct unit (Hz, not BPM)."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        if self.respiratory_rate <= 0:
            msg = f"The provided respiratory rate ({self.respiratory_rate:.2f}) must be positive."
            raise ValueError(msg)

        if self.heart_rate > UPPER_HEART_RATE_LIMIT:
            msg = (
                f"The provided heart rate ({self.heart_rate:.1f}) is higher "
                f"than {UPPER_HEART_RATE_LIMIT} Hz ({UPPER_HEART_RATE_LIMIT * MINUTE} BPM). "
                "Make sure this is correct, and to use the correct unit."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        if self.heart_rate <= 0:
            msg = f"The provided heart rate ({self.heart_rate:.2f}) must be positive."
            raise ValueError(msg)

        if self.respiratory_rate >= self.heart_rate:
            msg = (
                f"The respiratory rate ({self.respiratory_rate:.1f} Hz) is equal to or higher than the heart "
                f"rate ({self.heart_rate:.1f} Hz)."
            )
            raise ValueError(msg)

    @overload
    def apply(
        self, input_data: np.ndarray, sample_frequency: float, axis: int = 0, captures: dict | None = None
    ) -> np.ndarray: ...

    @overload
    def apply(self, input_data: ContinuousData, captures: dict | None = None, **kwargs) -> ContinuousData: ...

    @overload
    def apply(self, input_data: EITData, captures: dict | None = None, **kwargs) -> EITData: ...

    def apply(  # pyright: ignore[reportInconsistentOverload]
        self,
        input_data: T,
        sample_frequency: float | object = MISSING,
        axis: int | object = MISSING,
        captures: dict | None = None,
        **kwargs,
    ) -> T:
        """Filter data using multiple digital notch filters.

        Args:
            input_data: The data to filter. Can be a numpy array, ContinuousData, or EITData.
            sample_frequency:
                The sample frequency of the data. Should be provided when using a numpy array. If using
                ContinuousData or EITData, this will be taken from the data object.
            axis:
                The axis along which to apply the filter. Should only be provided when using a numpy array. Defaults to
                the first axis (0).
            captures:
                A dictionary to capture intermediate results for debugging or analysis. If provided, it will store the
                number of harmonics and the frequency bands used for filtering.
            **kwargs: Additional keyword arguments to pass to the ContinuousData or EITData object (e.g., `label`).
        """
        capture = make_capture(captures)
        capture("low_pass_frequency", self.noise_frequency_limit)
        capture("unfiltered_data", input_data)

        sample_frequency_, axis_, data = self._validate_arguments(
            input_data=input_data, sample_frequency=sample_frequency, axis=axis
        )

        # Ensure the data is filtered up to the point where lower_limit would be larger than the noise frequency limit
        n_harmonics = math.floor((self.noise_frequency_limit + self.notch_distance) / self.heart_rate)
        capture("n_harmonics", n_harmonics)

        for harmonic in range(1, n_harmonics + 1):
            data = self._filter_harmonic_with_bandstop(data, harmonic, axis_, sample_frequency_, capture)

        # Filter everything above noise limit
        sos = signal.butter(
            N=self.order,
            Wn=self.noise_frequency_limit,
            fs=sample_frequency_,
            btype="low",
            output="sos",
        )
        new_data = signal.sosfiltfilt(sos, data, axis_)

        if isinstance(input_data, np.ndarray):
            capture("filtered_data", new_data)
            return new_data

        # TODO: Replace with input_data.update(...) when implemented
        return_object = copy.deepcopy(input_data)
        for attr, value in kwargs.items():
            setattr(return_object, attr, value)

        if isinstance(return_object, ContinuousData):
            return_object.values = new_data
        elif isinstance(return_object, EITData):
            return_object.pixel_impedance = new_data

        capture("filtered_data", return_object)
        return return_object

    def _validate_arguments(
        self,
        input_data: np.ndarray | ContinuousData | EITData,
        sample_frequency: float | object,
        axis: int | object,
    ) -> tuple[float, int, np.ndarray]:
        if isinstance(input_data, ContinuousData | EITData):
            if sample_frequency is not MISSING:
                msg = "Sample frequency should not be provided when using ContinuousData or EITData."
                raise ValueError(msg)

            if axis is not MISSING:
                msg = "Axis should not be provided when using ContinuousData or EITData."
                raise ValueError(msg)

        if isinstance(input_data, ContinuousData):
            data = input_data.values
            sample_frequency_ = cast("float", input_data.sample_frequency)
            axis_ = 0
        elif isinstance(input_data, EITData):
            data = input_data.pixel_impedance
            sample_frequency_ = cast("float", input_data.sample_frequency)
            axis_ = 0
        elif isinstance(input_data, np.ndarray):
            data = input_data
            axis_ = DEFAULT_AXIS if axis is MISSING else axis
            axis_ = cast("int", axis_)
            if sample_frequency is MISSING:
                msg = "Sample frequency must be provided when using a numpy array."
                raise ValueError(msg)
            sample_frequency_: float = cast("float", sample_frequency)
        else:
            msg = f"Invalid input data type ({type(input_data)}). Must be a numpy array, ContinuousData, or EITData."
            raise TypeError(msg)

        if not sample_frequency_:
            msg = "Sample frequency must be provided."
            raise ValueError(msg)
        return sample_frequency_, axis_, data

    def _filter_harmonic_with_bandstop(
        self,
        data_: np.ndarray,
        harmonic: int,
        axis: int,
        sample_frequency: float,
        capture: _CaptureFunc,
    ) -> np.ndarray:
        lower_limit = self.heart_rate * harmonic - self.notch_distance
        upper_limit = self.heart_rate * harmonic + self.notch_distance

        if harmonic == 1:
            new_lower_limit = (self.heart_rate + self.respiratory_rate) / 2
            lower_limit = max(lower_limit, new_lower_limit)

        sos = signal.butter(
            N=self.order,
            Wn=[lower_limit, upper_limit],
            fs=sample_frequency,
            btype="bandstop",
            output="sos",
        )

        capture("frequency_bands", (lower_limit, upper_limit), append_to_list=True)

        return signal.sosfiltfilt(sos, data_, axis=axis)

    @property
    def plotting(self) -> FilterPlotting:
        """Return the plotting class for this filter."""
        from eitprocessing.plotting.filter import FilterPlotting

        return FilterPlotting()
