"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to when electrical impedance tomographs are read.
"""

import copy
import warnings
from dataclasses import dataclass
from dataclasses import field
import numpy as np
from IPython.display import HTML
from IPython.display import display
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm


@dataclass(eq=False)
class Frameset:
    """A Frameset is ....
        From Sequence:
            EIT data is contained within Framesets. A Frameset shares the time axis with a Sequence.
        From notebook:
            Data is contained within framesets.
            A frameset contains data that has been processed or edited in some way.
            By default, loaded data only contains raw impedance data, without any processing.
            Impedance data is available as the `pixel_values` attribute, which is a `np.ndarray` with three dimensions: time, rows, columns.
    Args:
        name (str): ...
        description (str): ...
        params (dict): ...
        pixel_values (np.ndarray): ...
        waveform_data (dict): ...
    """

    name: str
    description: str
    params: dict = field(default_factory=dict)
    pixel_values: np.ndarray = field(repr=False, default=None)
    waveform_data: dict = field(repr=False, default_factory=dict)

    def __len__(self):
        return self.pixel_values.shape[0]

    def __eq__(self, other):
        for attr in ["name", "description", "params"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        for attr in ["pixel_values"]:
            # TODO: why loop over a single attribute (here and below)?
            # Is it for future proofing

            # NaN values are not equal. Check whether values are equal or both NaN.
            s = getattr(self, attr)
            o = getattr(other, attr)
            if not np.all((s == o) | (np.isnan(s) & np.isnan(o))):
                # TODO: check that this works if isnan in different positions
                return False

        for attr in ["waveform_data"]:
            s = getattr(self, attr)
            o = getattr(other, attr)

            # check whether they contain the same types of data
            if set(s.keys()) != set(o.keys()):
                return False

            # NaN values are not equal. Check whether values are equal or both NaN
            for key, s_values in s.items():
                o_values = o[key]
                if not np.all(
                    (s_values == o_values) | (np.isnan(s_values) & np.isnan(o_values))
                ):
                    return False

        return True

    def select_by_indices(self, indices):
        # TODO: check https://stackoverflow.com/questions/47190218/proper-type-hint-for-getitem
        # TODO: check how this is done in sequence.py and follow same logic? Maybe even make external function called by both?
        obj = self.deepcopy()
        obj.pixel_values = self.pixel_values[indices, :, :]
        for key, values in self.waveform_data.items():
            obj.waveform_data[key] = values[indices]
        return obj

    __getitem__ = select_by_indices
    # TODO: consider directly defining getitem instead of taking extra step here

    @property
    def global_baseline(self):
        return np.nanmin(self.pixel_values)

    @property
    def pixel_values_global_offset(self):
        return self.pixel_values - self.global_baseline

    @property
    def pixel_baseline(self):
        return np.nanmin(self.pixel_values, axis=0)

    @property
    def pixel_values_individual_offset(self):
        return self.pixel_values - np.min(self.pixel_values, axis=0)

    @property
    def global_impedance(self):
        return np.nansum(self.pixel_values, axis=(1, 2))

    def plot_waveforms(self, waveforms=None):
        # TODO: document this function.
        if waveforms is None:
            waveforms = list(self.waveform_data.keys())

        n_waveforms = len(waveforms)
        fig, axes = plt.subplots(
            n_waveforms, 1, sharex=True, figsize=(8, 3 * n_waveforms)
        )
        fig.tight_layout()

        for ax, key in zip(axes, waveforms):
            ax.plot(self.waveform_data[key])
            ax.set_title(key)

    def animate(
        self,
        cmap: str = "plasma",
        show_progress: bool | str = "notebook",
        waveforms: bool | list[str] = False,
    ):  # pylint: disable = too-many-locals
        # TODO: document this function.
        # TODO: what would a bool do in show_progress? Should it be a Literal?
        if waveforms is True:
            waveforms = list(self.waveform_data.keys())

        if waveforms:
            n_waveforms = len(waveforms)
            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(1, 2, 1)
        else:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(1, 1, 1)

        array = self.pixel_values

        vmin = np.nanmin(array)
        vmax = np.nanmax(array)

        im = ax.imshow(array[0, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(im)

        if waveforms:
            wf_axes = []
            wf_lines = []
            last_wf_ax = None
            for n, key in enumerate(reversed(waveforms)):
                wf_ax = fig.add_subplot(n_waveforms, 2, 2 * (n + 1), sharex=last_wf_ax)
                wf_axes.append(wf_ax)
                if n == 0:
                    last_wf_ax = wf_ax

                wf_data = self.waveform_data[key][0]
                wf_lines.append(wf_ax.plot([0], wf_data))
                wf_ax.set_xlim((0, len(self)))
                wf_ax.set_ylim((wf_data.min(), wf_data.max()))

        if show_progress:
            if show_progress == "notebook":
                progress_bar = notebook_tqdm(total=len(self))
            else:
                progress_bar = tqdm(total=len(self))
            progress_bar.update(1)

        def update(i):
            im.set(data=array[i, :, :])

            if waveforms:
                for key, line in zip(waveforms, wf_lines):
                    line[0].set_xdata(range(i))
                    line[0].set_ydata(self.waveform_data[key][: i + 1])

            if show_progress:
                progress_bar.update(1)

        anim = animation.FuncAnimation(
            fig, update, frames=range(1, len(self)), repeat=False
        )
        display(HTML(anim.to_jshtml(self.params["framerate"])))

        plt.close()

    @classmethod
    def merge(cls, a, b):
        # TODO: see `merge` (and `__add__` functions in sequence)
        # docstring?
        if (a_ := a.name) != (b_ := b.name):
            raise ValueError(f"Frameset names don't match: {a_}, {b_}")

        if (a_ := a.description) != (b_ := b.description):
            raise ValueError(f"Frameset descriptions don't match: {a_}, {b_}")

        if (a_ := a.params) != (b_ := b.params):
            raise ValueError(f"Frameset params don't match: {a_}, {b_}")

        a_waveform_keys = set(a.waveform_data.keys())
        b_waveform_keys = set(b.waveform_data.keys())
        shared_waveform_keys = a_waveform_keys & b_waveform_keys
        not_shared_waveform_keys = a_waveform_keys ^ b_waveform_keys

        if len(not_shared_waveform_keys):
            warnings.warn(
                f"Some waveforms are not available in both framesets: {not_shared_waveform_keys}",
                UserWarning,
            )

        waveform_data = {}
        for key in shared_waveform_keys:
            waveform_data[key] = np.concatenate(
                [a.waveform_data[key], b.waveform_data[key]]
            )

        # for waveforms in a but not in b
        for key in a_waveform_keys - b_waveform_keys:
            b_values = np.full((len(b),), np.nan)
            waveform_data[key] = np.concatenate([a.waveform_data[key], b_values])

        # for waveforms in b but not in a
        for key in b_waveform_keys - a_waveform_keys:
            a_values = np.full((len(a),), np.nan)
            waveform_data[key] = np.concatenate([a_values, b.waveform_data[key]])

        return cls(
            name=a.name,
            description=a.description,
            params=a.params,
            pixel_values=np.concatenate([a.pixel_values, b.pixel_values], axis=0),
            waveform_data=waveform_data,
        )

    deepcopy = copy.deepcopy
