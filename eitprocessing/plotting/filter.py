from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from matplotlib import axes, figure
from matplotlib import pyplot as plt
from scipy import signal

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData

T = TypeVar("T", bound=np.ndarray | ContinuousData | EITData)

MINUTE = 60
MISSING = object()


@dataclass(frozen=True)
class FilterPlotting:
    """Utility class for plotting the effects of frequency filtering."""

    @classmethod
    def plot_results(
        cls,
        *,
        unfiltered_data: T,
        filtered_data: T,
        ax: axes.Axes | None = None,
        sample_frequency: float | object = MISSING,
        high_pass_frequency: float | None = None,
        low_pass_frequency: float | None = None,
        frequency_bands: list[tuple[float, float]] | None = None,
        xlim_to_max_filter_freq: float | None = 2,
        **kwargs,
    ) -> axes.Axes:
        """Plot frequency filtering results.

        This function is designed to work with the captures from a ButterworthFilter or MDNFilter.

        The plot shows the power spectra for the unfiltered and filtered data. If a high pass frequency or low pass
        frequency is provided, vertical lines are drawn at these frequencies. If frequency bands are provided, shaded
        regions are drawn for these bands.

        The provided data can be either numpy arrays, ContinuousData, or EITData, and must be the same for the
        unfiltered and filtered data. If numpy arrays are used, the sample frequency must be provided. If ContinuousData
        or EITData are used, the sample frequency is taken from the data itself.

        If an Axes instance is provided, the plot is drawn on that Axes. If not, a new figure and axes are created.

        Axes scaling:
            The x-axis and y-axis can be scaled linearly or logarithmically using the `xscale` and `yscale` arguments.
            If not provided, `xscale` is set to "linear" and `yscale` is set to "log". If the x-scale is logarithmic,
            the default limits are used.

            If the x-scale is linear, the upper limit is determined by the highest filter frequency or the sample
            frequency. If xlim_to_max_filter_freq is None, the x-axis ranges from 0 to half the sample frequency. If a
            value is provided, the x-axis limit is set to the maximum frequency of all filters multiplied by this value.
            This enables focussing on the filtered frequencies. Custom limits can be set using the returned axes object.

            If the y-scale is linear, the default limits are used. If the y-scale is logarithmic, the axes are scaled
            such that the upper 95% of the data falls within the middle 80% of the axes. This ensures that very small
            outliers don't disproportionally scale the y-axis.

        Warning:
            Although this function plots with the x-axis in beats per minute (BPM), the frequencies provided to it
            should be in Hz.

        Example:
        ```
        >>> mdn = MDNFilter(...)
        >>> captures = {}
        >>> filtered_data = mdn.apply(impedance_data, captures=captures)
        >>> ax = FilterResultsPlotting().plot(**captures)
        ```

        Args:
            unfiltered_data: The original data before filtering. Can be a numpy array, ContinuousData, or EITData.
            filtered_data: The data after filtering. Must be the same type as unfiltered_data.
            ax: Optional. A matplotlib Axes instance to plot on. If None, a new figure and axes will be created.
            sample_frequency:
                The sample frequency of the data. Must be provided only if input data are numpy arrays.
            frequency_bands:
                Optional. A list of tuples defining frequency bands to highlight in the plot. Each tuple contains (low,
                high) frequencies in Hz.
            high_pass_frequency:
                Optional. The critical frequency for a high pass filter in Hz.
            low_pass_frequency:
                Optional. The critical frequency for a low pass filter in Hz.
            xlim_to_max_filter_freq:
                Optional. If provided, the upper x-axis limit will be set to the maximum frequency of all filters, times
                this value. If None, the x-axis is limited to half the sample frequency.
            **kwargs:
                Extra keyword arguments to pass to the Axes.set() method, such as `xlabel`, `ylabel`, `title`, `yscale`,
                and `xscale`.

        Returns:
            A matplotlib Axes instance with the plot of the frequency filtering results.
        """
        unfiltered_signal, filtered_signal, sample_frequency_ = cls._get_data(
            unfiltered_data, filtered_data, sample_frequency
        )

        frequencies, unfiltered_power = signal.periodogram(unfiltered_signal, fs=sample_frequency_)
        _, filtered_power = signal.periodogram(filtered_signal, fs=sample_frequency_)

        fig, ax_ = cls._get_axes(ax)

        handles = []

        h_unfiltered = ax_.plot(frequencies * MINUTE, unfiltered_power, label="Unfiltered power", alpha=0.75)
        h_filtered = ax_.plot(frequencies * MINUTE, filtered_power, label="Filtered power", alpha=0.75)
        handles.extend([*h_unfiltered, *h_filtered])

        max_filtered_freq = cls._plot_frequency_annotations(
            high_pass_frequency, low_pass_frequency, frequency_bands, ax_, handles
        )

        ax_.legend(handles=handles, loc="upper right")

        yscale = kwargs.pop("yscale", "log")
        xscale = kwargs.pop("xscale", "linear")

        if yscale == "log":
            new_ylim_narrow = np.quantile(np.concatenate((unfiltered_power, filtered_power)), (0.05, 1))
            log_ylim = [np.log10(y) for y in new_ylim_narrow]
            range_ = log_ylim[1] - log_ylim[0]
            new_ylim = (10 ** (log_ylim[0] - 0.1 * range_), 10 ** (log_ylim[1] + 0.1 * range_))
            ax_.set(ylim=new_ylim)

        if xscale == "linear":
            if xlim_to_max_filter_freq is None:
                ax_.set(xlim=(0, sample_frequency_ / 2 * MINUTE))
            else:
                ax_.set(xlim=(0, max_filtered_freq * MINUTE * xlim_to_max_filter_freq))

        xlabel = kwargs.pop("xlabel", "Frequency (Hz)")
        ylabel = kwargs.pop("ylabel", "Power")
        title = kwargs.pop("title", "Frequency filtering results")

        for key in ["n_harmonics"]:
            kwargs.pop(key, None)

        ax_.set(yscale=yscale, xscale=xscale, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)

        if isinstance(fig, figure.Figure):
            fig.tight_layout()
        return ax_

    @classmethod
    def _get_axes(cls, ax: axes.Axes | None) -> tuple[figure.Figure | figure.SubFigure, axes.Axes]:
        if ax is None:
            fig, ax_ = plt.subplots(1, figsize=(6, 4))
        elif isinstance(ax, axes.Axes):
            ax_ = ax
            fig = ax_.figure
        else:
            msg = "The provided ax must be a matplotlib Axes instance."
            raise TypeError(msg)
        return fig, ax_

    @classmethod
    def _plot_frequency_annotations(
        cls,
        high_pass_frequency: float | None,
        low_pass_frequency: float | None,
        frequency_bands: list[tuple[float, float]] | None,
        ax_: axes.Axes,
        handles: list,
    ) -> float:
        high_filtered_freqs = []
        if frequency_bands is not None:
            h_vspan = None
            for low, high in frequency_bands:
                h_vspan = ax_.axvspan(low * MINUTE, high * MINUTE, fc="red", alpha=0.2, label="Filtered band(s)")
                high_filtered_freqs.append(high)

            if h_vspan is not None:
                handles.append(h_vspan)

        if high_pass_frequency is not None:
            h_line = ax_.axvline(high_pass_frequency * MINUTE, color="tab:purple", label="High-pass frequency")
            handles.append(h_line)
            high_filtered_freqs.append(high_pass_frequency)

        if low_pass_frequency is not None:
            lp_line = ax_.axvline(low_pass_frequency * MINUTE, color="tab:green", label="Low-pass frequency")
            handles.append(lp_line)
            high_filtered_freqs.append(low_pass_frequency)

        return max(high_filtered_freqs)

    @classmethod
    def _get_data(
        cls, unfiltered_data: T, filtered_data: T, sample_frequency: float | object
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if type(unfiltered_data) is not type(filtered_data):
            msg = "Unfiltered and filtered data must be of the same type."
            raise TypeError(msg)

        match unfiltered_data, filtered_data:
            case np.ndarray(), np.ndarray():
                unfiltered_signal = unfiltered_data
                filtered_signal = filtered_data
                sample_frequency_ = sample_frequency
            case ContinuousData(), ContinuousData():
                unfiltered_signal = unfiltered_data.values
                filtered_signal = filtered_data.values
                sample_frequency_ = unfiltered_data.sample_frequency
                if sample_frequency is not MISSING:
                    msg = "Sample frequency should not be provided when using ContinuousData."
                    raise ValueError(msg)
            case EITData(), EITData():
                unfiltered_signal = unfiltered_data.pixel_impedance
                filtered_signal = filtered_data.pixel_impedance
                sample_frequency_ = unfiltered_data.sample_frequency
                if sample_frequency is not MISSING:
                    msg = "Sample frequency should not be provided when using EITData."
                    raise ValueError(msg)
            case _:
                msg = "Unfiltered and filtered data must be either numpy arrays, ContinuousData, or EITData."
                raise TypeError(msg)

        if not isinstance(sample_frequency_, (float, int)):
            msg = "Sample frequency must be provided as float or int."
            raise TypeError(msg)
        return unfiltered_signal, filtered_signal, sample_frequency_
