# %%
import itertools

import numpy as np
from matplotlib import pyplot as plt

from eitprocessing.features.breath_detection import BreathDetection


def make_cosine_wave(sample_frequency: float, length: int, frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Create a time axis and cosine wave with given sample frequency, length, and frequency."""
    time = np.arange(length) / sample_frequency
    return time, np.cos(time * np.pi * 2 * frequency)


sample_frequency = 20
time, y = make_cosine_wave(sample_frequency, 300, 1)

plt.plot(time, y)

# %%


bd = BreathDetection()

start_invalid = np.argmax(time >= 3.25)
end_invalid = np.argmax(time >= 3.35)
y_with_invalid = np.copy(y)
y_with_invalid[start_invalid:end_invalid] = -100
y_with_invalid, outliers = bd._remove_invalid_data(y_with_invalid)

peak_valley_data = bd._detect_peaks_and_valleys(y, sample_frequency)
peak_valley_data2 = bd._detect_peaks_and_valleys(y_with_invalid, sample_frequency)
b2 = bd._create_breaths_from_peak_valley_data(time, peak_valley_data2.peak_indices, peak_valley_data2.valley_indices)
b2 = bd._remove_breaths_around_invalid_data(b2, y_with_invalid, time, sample_frequency, outliers)

plt.close("all")

plt.figure()
plt.plot(time, y)
color = itertools.cycle(["green", "red"])
for valleys, c in zip(itertools.pairwise(peak_valley_data.valley_indices), color, strict=False):
    plt.axvspan(time[valleys[0]], time[valleys[1]], color=c, alpha=0.3)


plt.figure()
plt.plot(time, y_with_invalid)
color = itertools.cycle(["green", "red"])
for valleys, c in zip(itertools.pairwise(peak_valley_data2.valley_indices), color, strict=False):
    plt.axvspan(time[valleys[0]], time[valleys[1]], color=c, alpha=0.3)

plt.figure()
plt.plot(time, y_with_invalid)
color = itertools.cycle(["green", "red"])
for breath, c in zip(b2, color, strict=False):
    plt.axvspan(breath.start_time, breath.end_time, color=c, alpha=0.3)

# %%


# %%


def ffill(data):
    while any(np.isnan(data[1:])):
        data = np.where(np.isnan(data), np.concatenate([data[:1], data[:-1]]), data)
    return data


def bfill(data):
    while any(np.isnan(data[:-1])):
        data = np.where(np.isnan(data), np.concatenate([data[1:], data[-1:]]), data)
    return data


# %%

data = y_with_invalid

data_mean = np.mean(data)

lower_percentile = np.percentile(data, 5)
cutoff_low = data_mean - (data_mean - lower_percentile) * 4

upper_percentile = np.percentile(data, 100 - 5)
cutoff_high = data_mean + (upper_percentile - data_mean) * 4

# create an array with 1s where there is an outlier
outliers = (data < cutoff_low) | (data > cutoff_high)
data[outliers] = np.nan
data = bfill(ffill(data))

plt.figure()
plt.plot(time, outliers.astype(int))
plt.plot(time, data)
plt.ylim((-2, None))

# %%

pvd = bd._detect_peaks_and_valleys(data, sample_frequency)
# %%
sample_frequency = 20
time, y = make_cosine_wave(sample_frequency, 1000, 1)
bd = BreathDetection(sample_frequency)

peak_valley_data = bd._detect_peaks_and_valleys(y, sample_frequency)

start_invalid = np.argmax(time >= 2.75)
end_invalid = np.argmax(time >= 3.25)
y_with_invalid = np.copy(y)
y_with_invalid[start_invalid:end_invalid] = np.nan
y_with_invalid = bd._fill_nan_with_nearest_neighbour(y_with_invalid)
data = y_with_invalid

data_mean = np.mean(data)

lower_percentile = np.percentile(data, 5)
cutoff_low = data_mean - (data_mean - lower_percentile) * 4

upper_percentile = np.percentile(data, 100 - 5)
cutoff_high = data_mean + (upper_percentile - data_mean) * 4

# create an array with 1s where there is an outlier
outliers = (data < cutoff_low) | (data > cutoff_high)
data[outliers] = np.nan
data = bfill(ffill(data))

plt.figure()
plt.plot(time, outliers.astype(int))
plt.plot(time, data)
plt.ylim((-2, None))
