from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.features.breath_detection import BreathDetection
from eitprocessing.parameters.eeli import EELI
from tests.test_breath_detection import _make_cosine_wave

MINUTE = 60


@pytest.fixture(scope="session")
def create_continuous_data_object():
    def internal(sample_frequency: float, duration: float, frequency: float) -> ContinuousData:
        time, values = _make_cosine_wave(sample_frequency, duration, frequency)
        return ContinuousData(
            "label",
            "name",
            "unit",
            "impedance",
            time=time,
            values=values,
            sample_frequency=sample_frequency,
        )

    return internal


@pytest.mark.parametrize("method", ["breath_detection"])
def test_init_succeed(method: str):
    _ = EELI(method=method)


@pytest.mark.parametrize("method", ["breathdetection", 1, None, ""])
def test_init_fail(method: Any):  # noqa: ANN401
    with pytest.raises(ValueError):
        _ = EELI(method=method)


@pytest.mark.parametrize(
    ("frequency", "duration", "sample_frequency"),
    [
        (10 / MINUTE, 2 * MINUTE, 20),
        (20 / MINUTE, MINUTE, 100),
        (1 / MINUTE, MINUTE, 10),
        (1 / MINUTE, 10 * MINUTE, 10),
    ],
)
def test_compute_parameter(
    create_continuous_data_object: Callable,
    frequency: float,
    duration: float,
    sample_frequency: float,
):
    expected_number_breaths = duration * frequency - 1
    cd = create_continuous_data_object(sample_frequency, duration * sample_frequency, frequency)
    eeli = EELI()
    eeli_values = eeli.compute_parameter(cd).values
    assert len(eeli_values) == expected_number_breaths
    if len(eeli_values) > 0:
        assert set(eeli_values.tolist()) == {-1.0}


@pytest.mark.parametrize("repeat_n", range(5))  # repeat this randomized test 5x
def test_eeli_values(repeat_n: int):  # noqa: ARG001
    # construct an array with 0s (peaks), interspersed with random negative numbers (valleys)
    n_valleys = np.random.default_rng().integers(50, 200, 1)
    valley_values = np.random.default_rng().integers(-100, -50, n_valleys)
    data = np.zeros(2 * n_valleys + 1)
    data[1::2] = valley_values
    sample_frequency = 1
    time = np.arange(len(data)) / sample_frequency

    expected_n_breaths = n_valleys - 1

    cd = ContinuousData(
        label="label",
        name="name",
        unit="unit",
        category="impedance",
        time=time,
        values=data,
        sample_frequency=sample_frequency,
    )
    with pytest.warns(DeprecationWarning):
        eeli = EELI(breath_detection_kwargs={"minimum_duration": 0})
    eeli_values = eeli.compute_parameter(cd).values

    assert len(eeli_values) == expected_n_breaths
    assert np.array_equal(eeli_values, valley_values[1:])


def test_bd_init():
    with pytest.warns(DeprecationWarning):
        assert EELI(breath_detection_kwargs={"minimum_duration": 0}) == EELI(
            breath_detection=BreathDetection(minimum_duration=0)
        )
    with pytest.warns(DeprecationWarning):
        EELI(breath_detection_kwargs={"minimum_duration": 0})
    with pytest.raises(TypeError):
        EELI(breath_detection_kwargs={"minimum_duration": 0}, breath_detection=BreathDetection(minimum_duration=0))


def test_with_data(draeger1: Sequence, pytestconfig: pytest.Config):
    if pytestconfig.getoption("--cov"):
        pytest.skip("Skip with option '--cov' so other tests can cover 100%.")

    cd = draeger1.continuous_data["global_impedance_(raw)"]
    eeli_values = EELI().compute_parameter(cd).values

    breaths = BreathDetection().find_breaths(cd)

    assert len(eeli_values) == len(breaths)


def test_non_impedance_data(draeger1: Sequence) -> None:
    cd = draeger1.continuous_data["global_impedance_(raw)"]
    original_category = cd.category

    _ = EELI().compute_parameter(cd)

    cd.category = "foo"
    with pytest.raises(ValueError):
        _ = EELI().compute_parameter(cd)

    cd.category = original_category
