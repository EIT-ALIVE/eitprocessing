import numpy as np
from matplotlib import pyplot as plt

from eitprocessing.eit_data.eit_data_variant import EITDataVariant

# TODO: remove line below to activate linting
# ruff: noqa


def animate_EITDataVariant(
    eit_data_variant: EITDataVariant,
    cmap: str = "plasma",
    show_progress: bool | str = "notebook",
    waveforms: bool | list[str] = False,
) -> None:
    # TODO: find other way to couple waveform data
    if waveforms is True:
        waveforms = list(eit_data_variant.waveform_data.keys())

    array = eit_data_variant.pixel_impedance

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

            wf_data = eit_data_variant.waveform_data[key][0]
            wf_lines.append(wf_ax.plot([0], wf_data))
            wf_ax.set_xlim((0, len(eit_data_variant)))
            wf_ax.set_ylim((wf_data.min(), wf_data.max()))

    else:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(array[0, :, :], vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar(im)

    if show_progress:
        if show_progress == "notebook":
            progress_bar = notebook_tqdm(total=len(eit_data_variant))
        else:
            progress_bar = tqdm(total=len(eit_data_variant))
        progress_bar.update(1)

    def update(i) -> None:
        im.set(data=array[i, :, :])

        if waveforms:
            for key, line in zip(waveforms, wf_lines, strict=False):
                line[0].set_xdata(range(i))
                line[0].set_ydata(eit_data_variant.waveform_data[key][: i + 1])

        if show_progress:
            progress_bar.update(1)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=range(1, len(eit_data_variant)),
        repeat=False,
    )
    display(HTML(anim.to_jshtml(eit_data_variant.params["framerate"])))

    plt.close()
