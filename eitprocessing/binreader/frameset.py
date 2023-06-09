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


@dataclass
class Frameset:
    name: str
    description: str
    params: dict = field(default_factory=dict)
    pixel_values: np.ndarray = field(repr=False, default=None)
    waveform_data: dict = field(repr=False, default_factory=dict)

    def __len__(self):
        return self.pixel_values.shape[0]

    def select_by_indices(self, indices):
        obj = copy.copy(self)
        obj.pixel_values = self.pixel_values[indices, :, :]
        for key in self.waveform_data.keys():
            obj.waveform_data[key] = self.waveform_data[key][indices]
        return obj

    __getitem__ = select_by_indices

    @property
    def global_baseline(self):
        return np.min(self.pixel_values)

    @property
    def pixel_values_global_offset(self):
        return self.pixel_values - self.global_baseline

    @property
    def pixel_baseline(self):
        return np.min(self.pixel_values, axis=0)

    @property
    def pixel_values_individual_offset(self):
        return self.pixel_values - np.min(self.pixel_values, axis=0)

    @property
    def global_impedance(self):
        return np.sum(self.pixel_values, axis=(1, 2))
    
    def plot_waveforms(self, waveforms=None):
        if waveforms is None:
            waveforms = list(self.waveform_data.keys())

        n_waveforms = len(waveforms)
        fig, axes = plt.subplots(n_waveforms, 1, sharex=True, figsize=(8, 3*n_waveforms))
        fig.tight_layout()

        for ax, key in zip(axes, waveforms):
            ax.plot(self.waveform_data[key])
            ax.set_title(key)

    def animate(self, cmap='plasma', show_progress='notebook', waveforms=False):

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
            wf_axes = list()
            wf_lines = list()
            last_wf_ax = None
            for n, key in enumerate(reversed(waveforms)):
                
                wf_ax = fig.add_subplot(n_waveforms, 2, 2*(n + 1), sharex=last_wf_ax)
                wf_axes.append(wf_ax)
                if n == 0:
                    last_wf_ax = wf_ax
                
                wf_data = self.waveform_data[key][0]
                wf_lines.append(wf_ax.plot([0], wf_data))
                wf_ax.set_xlim((0, len(self)))
                wf_ax.set_ylim((wf_data.min(), wf_data.max()))

        if show_progress:
            progress_bar = tqdm(total=len(self))
            if show_progress == 'notebook':
                progress_bar = notebook_tqdm(total=len(self))
            progress_bar.update(1)

        def update(i):
            im.set(data=array[i, :, :])
            
            if waveforms:
                for key, line in zip(waveforms, wf_lines):
                    line[0].set_xdata(range(i))
                    line[0].set_ydata(self.waveform_data[key][:i+1])

            if show_progress:
                progress_bar.update(1)

        anim = animation.FuncAnimation(fig, update, frames=range(1, len(self)), repeat=False)
        display(HTML(anim.to_jshtml(self.params['framerate'])))

        plt.close()

    @classmethod
    def merge(cls, a, b):
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
            warnings.warn(f"Some waveforms are not available in both framesets: {not_shared_waveform_keys}", UserWarning)

        waveform_data = dict()
        for key in shared_waveform_keys:
            waveform_data[key] = np.concatenate([a.waveform_data[key], b.waveform_data[key]])
        
        # for waveforms in a but not in b
        for key in a_waveform_keys - b_waveform_keys:
            b_values = np.full((len(b), ), np.nan)
            waveform_data[key] = np.concatenate([a.waveform_data[key], b_values])

        # for waveforms in b but not in a
        for key in b_waveform_keys - a_waveform_keys:
            a_values = np.full((len(a), ), np.nan)
            waveform_data[key] = np.concatenate([a_values, b.waveform_data[key]])

        return cls(
            name=a.name,
            description=a.description,
            params=a.params,
            pixel_values=np.concatenate([a.pixel_values, b.pixel_values], axis=0),
            waveform_data=waveform_data
        )

    deepcopy = copy.deepcopy
