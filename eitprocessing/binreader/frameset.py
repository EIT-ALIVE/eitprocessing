"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to when electrical impedance tomographs are read.
"""

import copy
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

    def __len__(self):
        return self.pixel_values.shape[0]

    def select_by_indices(self, indices):
        obj = copy.copy(self)
        obj.pixel_values = self.pixel_values[indices, :, :]
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

    def animate(self, cmap='plasma', show_progress='notebook'):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

        array = self.pixel_values

        vmin = array.min()
        vmax = array.max()

        im = ax.imshow(array[0, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(im)

        if show_progress:
            progress_bar = tqdm(total=len(self))
            if show_progress == 'notebook':
                progress_bar = notebook_tqdm(total=len(self))
            progress_bar.update(1)

        def update(i):
            im.set(data=array[i, :, :])
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

        return cls(
            name=a.name,
            description=a.description,
            params=a.params,
            pixel_values=np.concatenate([a.pixel_values, b.pixel_values], axis=0),
        )

    deepcopy = copy.deepcopy
