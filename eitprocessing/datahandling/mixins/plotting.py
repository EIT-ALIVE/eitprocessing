from __future__ import annotations

from matplotlib import pyplot as plt

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.sequence import Sequence


class Plotting:
    """Mixin class that adds methods for graphical outputs."""

    def plot_waveforms(
        self: Sequence | ContinuousData,
        waveforms: str | list[str],  # TODO: check that str is correct type hint
        reset_x: bool = False,
    ) -> None:
        """Plot waveform data against time.

        Args:
            waveforms: Name(s) of ContinuousData objects to plot. #TODO: can this be other than ContinuousData?
            reset_x:
                If False (default), the time (x-) axis starts at t0 (i.e., the first time point in the `Sequence`).
                If True, the time (x-) axis starts at 0 (i.e. t0 is subtracted from each time point).
        """
        if not isinstance(self, (Sequence | ContinuousData)):
            msg = "XXXXXXXX"  # TODO: write error message
            raise TypeError(msg)

        if not isinstance(waveforms, list):
            waveforms = [waveforms]
        n_waveforms = len(waveforms)

        offset = self.time[0] if reset_x else 0

        fig, axes = plt.subplots(n_waveforms, 1, sharex=True, figsize=(8, 3 * n_waveforms))
        fig.tight_layout()

        for ax, key in zip(axes, waveforms, strict=False):
            ax.plot(self.time - offset, self.waveform_data[key])
            ax.set_title(key)
