from collections.abc import Callable

import numpy as np
import pytest

from eitprocessing.datahandling.eitdata import EITData, Vendor
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.roi import PixelMask
from eitprocessing.roi.watershed import WatershedLungspace


@pytest.fixture
def simulated_eit_data():
    """Fixture to create a simulated EITData object for testing."""

    def factory(
        breath_duration: float = 4.5,  # s
        sample_frequency: float = 20,  # Hz
        size: int = 32,  # Size of the square grid
        n_peaks: int = 2,  # Number of peaks in a single direction; actual number of peaks is the square of this value
        relative_phase_shift: float = 0.5,  # Relative phase shift for the wave pattern
        amplitude_clip_shift: float = 0.5,  # Determines how much of the signal has amplitude. 0 => 50%, 1 => 100%.
        end_expiratory_value: float = 0.01,  # Value at the end of expiration, to prevent all values going to 0
        base_value: float = 0.8,  # Base value for the sinusoid shape, so most signal is >0
    ) -> EITData:
        time = np.arange(30 * sample_frequency) / sample_frequency

        # Swap x and y to match the row/column convention instead of x/y convention
        y_grid, x_grid = np.indices((size, size))
        t_grid = time[:, None, None]

        # Create a phase shift that increases linearly with x and y position
        phase_shift = (x_grid + y_grid) / size * np.pi * relative_phase_shift

        amplitude = (
            np.clip(np.sin(t_grid * 2 * np.pi / breath_duration - phase_shift) + amplitude_clip_shift, 0, None)
            / (1 + amplitude_clip_shift)
            * ((0.2 * size + (size - x_grid)) / (1.2 * size) * (0.5 * size + (size - y_grid)) / (1.5 * size))
        )
        amplitude = amplitude / np.max(amplitude) * (1 - end_expiratory_value)

        sinusoid_shape = (
            np.sin((x_grid - size / 8) / size * 2 * np.pi * n_peaks)  # variation in x-direction
            + np.sin((y_grid - size / 8) / size * 2 * np.pi * n_peaks)  # variation in y-direction
            + base_value  # base value, so most signal is >0
        )
        sinusoid_shape = sinusoid_shape / np.max(sinusoid_shape)

        pixel_impedance = sinusoid_shape * amplitude + end_expiratory_value
        return EITData(
            pixel_impedance=pixel_impedance,
            path="",
            nframes=len(time),
            time=time,
            sample_frequency=sample_frequency,
            vendor=Vendor.SIMULATED,
            label="simulated",
            description="Simulated EIT data for testing purposes",
            name="",
            suppress_simulated_warning=True,
        )

    return factory


def test_watershed_init():
    _ = WatershedLungspace(threshold_fraction=0.2)
    _ = WatershedLungspace()


@pytest.mark.parametrize("threshold", [-0.2, 0.0, 1.0, 1.2])
def test_watershed_init_threshold_outside_range(threshold: float):
    with pytest.raises(ValueError, match="Threshold must be between 0 and 1."):
        _ = WatershedLungspace(threshold_fraction=threshold)


@pytest.mark.parametrize("threshold", ["0.2", True, None])
def test_watershed_init_threshold_wrong_type(threshold: object):
    with pytest.raises(TypeError, match="Threshold must be a float."):
        _ = WatershedLungspace(threshold_fraction=threshold)


@pytest.mark.parametrize("threshold", [0.1, 0.15, 0.2, 0.25, 0.3])
def test_watershed_with_real_data(draeger_50hz_healthy_volunteer_pressure_pod: Sequence, threshold: float):
    eit_data = draeger_50hz_healthy_volunteer_pressure_pod.eit_data["raw"]
    watershed_mask = WatershedLungspace(threshold_fraction=threshold).apply(eit_data, captures=(captures := {}))

    assert captures, "captures should have values after runnning apply"
    assert isinstance(watershed_mask, PixelMask)
    assert np.nansum(watershed_mask.values) > 0
    assert len(captures["local peaks"])  # There should be a non-empty list of local peaks

    tiv_mask = captures["functional tiv mask"]
    assert isinstance(tiv_mask, PixelMask)

    amplitude_mask = captures["functional amplitude mask"]
    assert isinstance(amplitude_mask, PixelMask)

    assert np.nansum((tiv_mask - amplitude_mask).mask) == 0, "TIV mask should be a subset of amplitude mask"
    assert np.nansum((tiv_mask - watershed_mask).mask) == 0, "TIV mask should be a subset of the watershed mask"
    assert np.nansum((watershed_mask - amplitude_mask).mask) == 0, "TIV mask should be a subset of the watershed mask"

    included_pixels = (~np.isnan(captures["mean tiv"].values)) & (~np.isnan(captures["mean amplitude"].values))
    assert np.all(captures["mean tiv"].values <= captures["mean amplitude"].values, where=included_pixels)


@pytest.mark.parametrize("threshold", [0.1, 0.15, 0.2, 0.25, 0.3])
def test_watershed_with_simulated_data(simulated_eit_data: Callable[..., EITData], threshold: float):
    """Test the WatershedLungspace with simulated EIT data."""
    eit_data = simulated_eit_data()
    watershed_mask = WatershedLungspace(threshold_fraction=threshold).apply(eit_data, captures=(captures := {}))

    assert isinstance(watershed_mask, PixelMask)
    assert len(captures["local peaks"]), "There should be a non-empty list of local peaks"
    assert np.nansum(watershed_mask.values) > 0, "The watershed mask should have some values"

    tiv_mask = captures["functional tiv mask"]
    amplitude_mask = captures["functional amplitude mask"]
    assert isinstance(tiv_mask, PixelMask)
    assert isinstance(amplitude_mask, PixelMask)

    assert np.nansum((tiv_mask - amplitude_mask).mask) == 0, "TIV mask should be a subset of the amplitude mask"
    assert np.nansum((tiv_mask - watershed_mask).mask) == 0, "TIV mask should be a subset of the watershed mask"
    assert np.nansum((watershed_mask - amplitude_mask).mask) == 0, "TIV mask should be a subset of the watershed mask"

    included_pixels = (~np.isnan(captures["mean tiv"].values)) & (~np.isnan(captures["mean amplitude"].values))
    assert np.all(captures["mean tiv"].values <= captures["mean amplitude"].values, where=included_pixels)


def test_watershed_timing_data(draeger_50hz_healthy_volunteer_pressure_pod: Sequence):
    eit_data = draeger_50hz_healthy_volunteer_pressure_pod.eit_data["raw"]
    timing_data = draeger_50hz_healthy_volunteer_pressure_pod.continuous_data["global_impedance_(raw)"]
    watershed_mask_w_timing = WatershedLungspace().apply(eit_data, timing_data=timing_data)
    watershed_mask = WatershedLungspace().apply(eit_data)

    assert watershed_mask_w_timing == watershed_mask, "The timing data should be the same in this case"


def test_watershed_captures(draeger_50hz_healthy_volunteer_pressure_pod: Sequence):
    eit_data = draeger_50hz_healthy_volunteer_pressure_pod.eit_data["raw"]

    with pytest.raises(TypeError):
        _ = WatershedLungspace().apply(eit_data, captures=(captures := []))

    _ = WatershedLungspace().apply(eit_data, captures=(captures := {}))
    assert captures, "captures should have values after runnning apply"
    for key in [
        "mean tiv",
        "functional tiv mask",
        "mean amplitude",
        "functional amplitude mask",
        "local peaks",
        "included marker indices",
        "included region",
        "watershed regions",
        "included peaks",
        "excluded peaks",
        "included watershed regions",
    ]:
        assert key in captures, f"captures should have a '{key}' entry"

    assert len(captures["local peaks"]) == len(captures["included peaks"]) + len(captures["excluded peaks"])


def test_watershed_no_amplitude():
    eit_data = EITData(
        pixel_impedance=np.ones((100, 32, 32)),
        path="",
        nframes=100,
        time=np.arange(100) / 20,
        sample_frequency=20,
        vendor=Vendor.SIMULATED,
        label="simulated",
        description="Simulated EIT data with no amplitude",
        name="",
        suppress_simulated_warning=True,
    )
    with pytest.raises(ValueError, match="No breaths found in TIV or amplitude data"):
        _ = WatershedLungspace().apply(eit_data)
