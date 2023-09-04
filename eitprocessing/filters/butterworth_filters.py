from scipy import signal
from . import TimeDomainFilter


MIN_ORDER = 1
MAX_ORDER = 10
FILTER_TYPES = ["lowpass", "highpass", "bandpass", "bandstop"]


class ButterworthFilter(TimeDomainFilter):
    """Butterworth Filter"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        filter_type: str,
        cutoff_frequency: float | tuple[float, float],
        order: int,
        sample_frequency: float,
        override_order: bool = False,
    ):
        if filter_type not in FILTER_TYPES:
            raise ValueError(
                f"The filter type should be one of {', '.join(FILTER_TYPES)}, not {filter_type}."
            )
        self.filter_type = filter_type

        if filter_type in ("lowpass", "highpass"):
            if not isinstance(cutoff_frequency, (int, float)):
                raise TypeError("`cutoff_frequency` should be an integer or float")

        elif filter_type in ("bandpass", "bandstop"):
            if not isinstance(cutoff_frequency, tuple):
                if isinstance(cutoff_frequency, str):
                    raise TypeError("`cutoff_frequency` should be a tuple")
                try:
                    cutoff_frequency = tuple(cutoff_frequency)
                except TypeError as e:
                    raise TypeError(
                        "`cutoff_frequency` should be (castable to) a tuple"
                    ) from e

            if len(cutoff_frequency) != 2:
                raise ValueError("`cutoff_frequency` should have length 2")

            if not all(isinstance(value, (int, float)) for value in cutoff_frequency):
                raise TypeError(
                    "`cutoff_frequency` should be a tuple containing two numbers"
                )
        
        self.cutoff_frequency = cutoff_frequency

        if order < MIN_ORDER or (order > MAX_ORDER and override_order is False):
            raise AttributeError(
                f"Order should be between {MIN_ORDER} and {MAX_ORDER}. "
                + "To use higher values, set `override_order` to `True`."
            )
        self.order = order

        if not isinstance(sample_frequency, (int, float)):
            raise TypeError('`sample_frequency` should be a positive number')
        if sample_frequency <= 0:
            raise ValueError('`sample_frequency` should be positive')
        self.sample_frequency = sample_frequency


    def __eq__(self, other):
        if not isinstance(other, ButterworthFilter):
            return False
        return self.__dict__ == other.__dict__

    def apply_filter(self, input_data):
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
    available_in_gui = False

    def __init__(self, *args, **kwargs):
        if "filter_type" in kwargs and kwargs["filter_type"] != self.filter_type:
            raise AttributeError("`filter_type` should not be supplied.")

        ButterworthFilter.__init__(self, *args, filter_type=self.filter_type, **kwargs)


class LowPassFilter(SpecifiedButterworthFilter):
    available_in_gui = True
    filter_type = "lowpass"


class HighPassFilter(SpecifiedButterworthFilter):
    available_in_gui = True
    filter_type = "highpass"


class BandStopFilter(SpecifiedButterworthFilter):
    available_in_gui = True
    filter_type = "bandstop"


class BandPassFilter(SpecifiedButterworthFilter):
    available_in_gui = True
    filter_type = "bandpass"
