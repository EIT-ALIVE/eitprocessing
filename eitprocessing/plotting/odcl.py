from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from eitprocessing.parameters.odcl import OdclPeepTrialResults


@dataclass(frozen=True)
class OdclPeepTrialPlotting:
    """Plotting functionality for OdclPeepTrialResults."""

    results: OdclPeepTrialResults

    def plot_overdistention_collapse(
        self,
        *,
        ax: Axes | None = None,
        labels: list[str] | None = None,
        xlabel: str | None = None,
    ) -> tuple[Figure, Axes]:
        """Plot overdistention and collapse values as line plots.

        If an axis is provided, the plots are created there. Otherwise, a new figure and axis is created.

        Labels can be provided for the steps, which will be shown as tick labels.
        The label for the x-axis can also be provided.

        Returns the figure and axis.
        """
        if ax is None:
            fig, ax = plt.subplots(layout="constrained")
        else:
            fig = None

        x = np.arange(1, len(self.overdistention) + 1)
        ax.plot(
            x,
            self.overdistention,
            label="Overdistention",
            marker="o",
            lw=3,
            color="darkorange",
        )
        ax.plot(x, self.collapse, label="Collapse", marker="s", lw=3, color="#707070")
        ax.set(xticks=x)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid()
        if fig:
            fig.legend(loc="outside upper center", ncol=2, frameon=False)
        else:
            ax.legend(loc="upper center", ncol=2)

        ndigits = int(np.log10(len(x))) + 1
        if labels is None:
            labels = [f"#{x_:0{ndigits}d}" for x_ in x]

        xlabel = xlabel or "PEEP steps (decremental)"

        ax.set(xticklabels=labels, xlabel=xlabel)
        return fig, ax

    def plot_odcl_maps(
        self,
        *,
        labels: list[str] | None = None,
        ncols: int = 4,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """Create plots of the ODCL maps.

        Generates a n_rows x n_cols grid of axes fitting all PEEP steps. Provided labels are set as axis titles.

        Returns the figure and axes.
        """
        n_steps = len(self.odcl_maps)
        n_rows = int(np.ceil(n_steps / ncols))
        fig, axes = plt.subplots(
            n_rows,
            ncols,
            layout="compressed",
            sharex=True,
            sharey=True,
            figsize=figsize or (2 * ncols + 1, 2.4 * n_rows),
        )

        for axis, map_, od, cl in zip(
            axes.flatten(),
            self.odcl_maps,
            self.overdistention,
            self.collapse,
            strict=False,
        ):
            map_.plotting.imshow(ax=axis)
            axis.set_xlabel(f"OD: {od:.0%}, CL: {cl:.0%}")

        n_digits = int(np.log10(n_steps)) + 1
        if labels is None:
            labels = [f"#{x + 1:0{n_digits}d}" for x in range(n_steps)]
        for axis, label in zip(axes.flatten(), labels, strict=False):
            axis.set(title=label)

        images = [axis.images[0] for axis in axes.flat if axis.images]
        old_colorbars = [image.colorbar for image in images if image.colorbar is not None]
        old = old_colorbars[0]

        fig.colorbar(
            images[0],
            ax=axes,
            format=old.formatter,
            extend=old.extend,
            label=old.ax.get_ylabel(),
        )

        for colorbar in old_colorbars:
            colorbar.remove()

        for axis in axes.flatten()[len(self.odcl_maps) :]:
            axis.remove()

        return fig, axes
