from typing import overload

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.ticker import MaxNLocator

from eitprocessing.features.rate_detection import RateDetection

MINUTE = 60


class RateDetectionPlotting:
    """Utility class to plot the results of the RateDetection algorithm.

    This class can be accessed through the `plotting` attribute of the RateDetection class.

    Example:
    ```python
    >>> rd = RateDetection("adult")
    >>> captures = {}
    >>> rr, hr = rd.apply(eit_data, captures=captures)
    >>> fig = rd.plotting(**captures)
    ```
    """

    def __init__(self, obj: RateDetection):
        self.obj = obj

    @overload
    def plot(
        self,
        *,
        frequencies: np.ndarray,
        normalized_total_power: np.ndarray,
        average_normalized_pixel_power: np.ndarray,
        diff_total_averaged_power: np.ndarray,
        estimated_respiratory_rate: float,
        estimated_heart_rate: float,
        fig: None,
    ) -> Figure: ...

    @overload
    def plot(
        self,
        *,
        frequencies: np.ndarray,
        normalized_total_power: np.ndarray,
        average_normalized_pixel_power: np.ndarray,
        diff_total_averaged_power: np.ndarray,
        estimated_respiratory_rate: float,
        estimated_heart_rate: float,
        fig: SubFigure,
    ) -> SubFigure: ...

    @overload
    def plot(
        self,
        *,
        frequencies: np.ndarray,
        normalized_total_power: np.ndarray,
        average_normalized_pixel_power: np.ndarray,
        diff_total_averaged_power: np.ndarray,
        estimated_respiratory_rate: float,
        estimated_heart_rate: float,
        fig: Figure,
    ) -> Figure: ...

    def plot(
        self,
        *,
        frequencies: np.ndarray,
        normalized_total_power: np.ndarray,
        average_normalized_pixel_power: np.ndarray,
        diff_total_averaged_power: np.ndarray,
        estimated_respiratory_rate: float,
        estimated_heart_rate: float,
        fig: Figure | SubFigure | None = None,
    ) -> Figure | SubFigure:
        """Plot the results of the RateDetection algorithm.

        This method is intended to be used after running the `detect_respiratory_heart_rate` method of the
        `RateDetection` class. It visualizes the frequency analysis results, including the normalized power
        spectrum, average normalized pixel spectrum, and the difference between the two.

        NB: Although the frequencies, rate limits and estimated rates are provided in Hz, they are visualized in
        breaths/beats per minute (bpm).

        Arguments:
            frequencies: Frequencies used in the analysis, in Hz.
            normalized_total_power: Normalized total power spectrum.
            average_normalized_pixel_power: Average normalized pixel power spectrum.
            diff_total_averaged_power:
                Difference between the normalized summed power spectra and the normalized total power.
            estimated_respiratory_rate: Estimated respiratory rate in Hz.
            estimated_heart_rate: Estimated heart rate in Hz.
            fig: Optional matplotlib Figure or SubFigure to plot on. If None, a new figure is created.

        Returns:
            A matplotlib figure or subfigure (if provided) containing the plots of the frequency analysis.
        """
        if fig is None:
            fig = plt.figure(figsize=(6, 4.5))

        axes = fig.subplots(3, 1, sharex=True)
        frequency_range = frequencies < self.obj.max_heart_rate * 1.25

        axes[0].set(ymargin=0.2)
        axes[0].plot(
            frequencies[frequency_range] * MINUTE,
            normalized_total_power[frequency_range],
            label="Normalized power",
            color="k",
        )
        axes[1].plot(
            frequencies[frequency_range] * MINUTE,
            average_normalized_pixel_power[frequency_range],
            label="Average normalized power",
            color="k",
        )
        axes[2].plot(
            frequencies[frequency_range] * MINUTE,
            diff_total_averaged_power[frequency_range],
            color="k",
            label="Difference",
        )

        ylim0 = axes[0].get_ylim()
        axes[0].fill_betweenx(
            ylim0,
            self.obj.min_respiratory_rate * MINUTE,
            self.obj.max_respiratory_rate * MINUTE,
            color="k",
            alpha=0.2,
            label="RR range",
        )
        axes[0].set(ylim=ylim0)
        axes[0].axvline(estimated_respiratory_rate * MINUTE, color="r", label="Estimated RR", zorder=1.9, lw=1.5)

        ylim2 = axes[2].get_ylim()
        axes[2].fill_betweenx(
            ylim2,
            self.obj.min_heart_rate * MINUTE,
            self.obj.max_heart_rate * MINUTE,
            color="k",
            alpha=0.2,
            label="HR range",
        )
        axes[2].set(ylim=ylim2)
        axes[2].axvline(estimated_heart_rate * MINUTE, color="r", label="Estimated HR", zorder=1.9, lw=1.5)

        axes[0].set(ylabel="Power (a.u.)", xmargin=0)
        axes[1].set(ylabel="Power (a.u.)")
        axes[1].sharey(axes[0])
        axes[2].set(ylim=(-0.2 * ylim2[1], ylim2[1]), ylabel="Diff. (a.u.)", xlabel="Rate (bpm)")

        axes[0].legend(loc="upper right")
        axes[1].legend(loc="upper right")
        axes[2].legend(loc="upper right")

        for axis in fig.axes:
            axis.xaxis.set_major_locator(MaxNLocator(nbins=15))

        axis0_top_x = axes[0].twiny()
        axis0_top_x.set(
            ylim=axes[0].get_ylim(),
            xlim=axes[0].get_xlim(),
            xticks=[estimated_respiratory_rate * MINUTE],
            xticklabels=[f"{estimated_respiratory_rate * MINUTE:.1f} bpm"],
        )

        axis2_top_x = axes[2].twiny()
        axis2_top_x.set(
            ylim=axes[2].get_ylim(),
            xlim=axes[2].get_xlim(),
            xticks=[estimated_heart_rate * MINUTE],
            xticklabels=[f"{estimated_heart_rate * MINUTE:.1f} bpm"],
        )

        for axis in (axis0_top_x, axis2_top_x):
            axis.tick_params(axis="x", length=0)
            for label in axis.get_xticklabels():
                label.set_color("red")

        fig.suptitle("Rate Detection Results")
        fig.align_ylabels(axes)

        if isinstance(fig, Figure):  # SubFigures do not have tight_layout
            fig.tight_layout()

        return fig
