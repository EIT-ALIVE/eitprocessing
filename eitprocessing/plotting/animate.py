import numpy as np
from IPython.display import HTML
from IPython.display import display
from matplotlib import animation
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm import tqdm_notebook
from eitprocessing.eit_data.eit_data_variant import EITDataVariant


def animate_pixel_impedance(
    array: np.ndarray,
    cmap: str = "plasma",
    show_progress: bool | str = "notebook",
    waveforms: bool | list[str] = False,
    framerate: float = 20,
):  # pylint: disable = too-many-locals
    # TODO: find other way to couple waveform data

    tqdm_ = tqdm_notebook if show_progress == "notebook" else tqdm

    if waveforms is True:
        waveforms = list(array.waveform_data.keys())

    vmin = np.nanmin(array)
    vmax = np.nanmax(array)

    if waveforms:
        n_waveforms = len(waveforms)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 2, 1)
        wf_axes = []
        wf_lines = []
        last_wf_ax = None
        for n, key in enumerate(reversed(waveforms)):
            wf_ax = fig.add_subplot(n_waveforms, 2, 2 * (n + 1), sharex=last_wf_ax)
            wf_axes.append(wf_ax)
            if n == 0:
                last_wf_ax = wf_ax

            wf_data = array.waveform_data[key][0]
            wf_lines.append(wf_ax.plot([0], wf_data))
            wf_ax.set_xlim((0, len(array)))
            wf_ax.set_ylim((wf_data.min(), wf_data.max()))

    else:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(array[0, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(im)

    progress_bar = None
    if show_progress:
        progress_bar = tqdm_(total=len(array))
        progress_bar.update(1)

    def update(i):
        im.set(data=array[i, :, :])

        if waveforms:
            for key, line in zip(waveforms, wf_lines):
                line[0].set_xdata(range(i))
                line[0].set_ydata(array.waveform_data[key][: i + 1])

        if show_progress:
            progress_bar.update(1)

    anim = animation.FuncAnimation(
        fig, update, frames=range(1, len(array)), repeat=False
    )
    display(HTML(anim.to_jshtml(int(framerate))))

    plt.close()
