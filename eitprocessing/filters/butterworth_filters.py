from scipy import signal
from . import TimeDomainFilter


class ButterworthFilter(TimeDomainFilter):
    """Butterworth Filter"""

    def __init__(
        self,
        filter_type: str,
        cutoff_frequency: float | tuple[float, float],
        order: int,
        sample_frequency: float,
        override_order: bool = False,
    ):
        self.cutoff_frequency = cutoff_frequency

        if order < 1 or (order > 10 and override_order is False):
            raise AttributeError(
                "Order should be an integer between 1 and 10. "
                + "To use higher values, set `override_order` to `True`."
            )

        self.order = order
        self.sample_frequency = sample_frequency
        self.filter_type = filter_type

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
