def plot_waveforms(self, waveforms=None):
    if waveforms is None:
        waveforms = list(self.waveform_data.keys())

    n_waveforms = len(waveforms)
    fig, axes = plt.subplots(n_waveforms, 1, sharex=True, figsize=(8, 3 * n_waveforms))
    fig.tight_layout()

    for ax, key in zip(axes, waveforms):
        ax.plot(self.waveform_data[key])
        ax.set_title(key)
