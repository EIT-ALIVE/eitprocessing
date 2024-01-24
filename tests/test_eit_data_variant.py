import itertools
from dataclasses import dataclass
from dataclasses import field
import numpy as np
import pytest
from typing_extensions import Self
from eitprocessing.eit_data.eit_data_variant import EITDataVariant
from eitprocessing.helper import NotEquivalent
from eitprocessing.mixins.slicing import SelectByIndex
from eitprocessing.variants import Variant


@pytest.fixture
def gen_pixel_impedance():
    def _pixel_impedance(length: int, baseline: float, amplitude: float):
        rng = np.random.default_rng()
        pixel_impedance = rng.random((length, 32, 32), np.float_) * amplitude + baseline

        # add some nan values
        for j, k, l in itertools.product(
            rng.integers(length, size=4),
            rng.integers(31, size=4),
            rng.integers(31, size=4),
        ):
            pixel_impedance[j, k, l] = np.nan

        # force the lowest value to be equal to baseline
        pixel_impedance = np.where(
            pixel_impedance == np.nanmin(pixel_impedance), baseline, pixel_impedance
        )
        return pixel_impedance

    return _pixel_impedance


@pytest.fixture
def NonEITDataVariant():
    @dataclass
    class NonEITDataVariant(Variant, SelectByIndex):
        pixel_impedance: np.ndarray = field(repr=False, kw_only=True)

        def _sliced_copy(
            self, start_index: int, end_index: int, label: str | None = None
        ) -> Self:
            return super()._sliced_copy(start_index, end_index, label)

        def concatenate(self: Self, other: Self) -> Self:
            return super().concatenate(other)

    return NonEITDataVariant


def test_init():
    _ = EITDataVariant("name", "label", "desc", pixel_impedance=np.zeros((100, 32, 32)))
    with pytest.raises(TypeError):
        _ = EITDataVariant("name", "label", "desc")

    with pytest.raises(ValueError):
        _ = EITDataVariant(
            "name", "label", "desc", pixel_impedance=np.zeros((0, 32, 32))
        )

    with pytest.raises(ValueError):
        _ = EITDataVariant(
            "name", "label", "desc", pixel_impedance=np.zeros((32, 32, 100))
        )

    with pytest.raises(ValueError):
        _ = EITDataVariant(
            "name", "label", "desc", pixel_impedance=np.zeros((100, 32, 16))
        )


def test_len(gen_pixel_impedance):
    for n in range(10, 100, 10):
        pixel_impedance = gen_pixel_impedance(n, 1, 1)
        edv = EITDataVariant(
            "name", "label", "description", pixel_impedance=pixel_impedance
        )
        assert len(edv) == pixel_impedance.shape[0]


def test_eq(gen_pixel_impedance, NonEITDataVariant):
    pixel_impedance = gen_pixel_impedance(1000, 0, 100)

    edv1 = EITDataVariant(
        "name",
        "label",
        "description",
        {"foo": "bar", "other": 1},
        pixel_impedance=pixel_impedance,
    )
    edv2 = EITDataVariant(
        "name",
        "label",
        "description",
        {"other": 1, "foo": "bar"},
        pixel_impedance=np.copy(pixel_impedance),
    )
    nedv = NonEITDataVariant(
        "name",
        "label",
        "description",
        {"foo": "bar", "other": 1},
        pixel_impedance=pixel_impedance,
    )

    assert edv1 == edv2

    edv2.name = "different_label"
    assert edv1 != edv2
    edv2.name = edv1.name

    edv2.params["other"] = 2
    assert edv1 != edv2
    edv2.params["other"] = edv1.params["other"]

    assert edv1 != nedv


def test_equivalent(gen_pixel_impedance):
    edv1 = EITDataVariant(
        "name",
        "label",
        "description",
        {"foo": "bar", "other": 1},
        pixel_impedance=gen_pixel_impedance(1, 0, 100),
    )
    edv2 = EITDataVariant(
        "name",
        "label",
        "description",
        {"other": 1, "foo": "bar"},
        pixel_impedance=gen_pixel_impedance(2, 0, 100),
    )

    assert edv1.check_equivalence(edv2)
    assert EITDataVariant.check_equivalence(edv1, edv2)
    edv1.check_equivalence(edv2, raise_=True)

    edv2.name = "different label"
    assert not edv1.check_equivalence(edv2)
    with pytest.raises(NotEquivalent):
        edv1.check_equivalence(edv2, raise_=True)
    edv2.name = edv1.name

    edv2.params["other"] = 2
    assert not EITDataVariant.check_equivalence(edv1, edv2)
    edv2.params["other"] = edv1.params["other"]


@pytest.mark.parametrize("execution_number", range(5))
def test_properties(gen_pixel_impedance, execution_number):
    rng = np.random.default_rng()
    baseline = rng.integers(0, 100)
    pixel_impedance = gen_pixel_impedance(1000, baseline, 100)
    edv1 = EITDataVariant(
        "name",
        "label",
        "description",
        {"foo": "bar", "other": 1},
        pixel_impedance=pixel_impedance,
    )

    assert edv1.global_baseline == baseline
    assert np.nanmin(edv1.pixel_impedance_global_offset) == 0
    assert np.array_equal(
        edv1.pixel_baseline, np.nanmin(pixel_impedance, axis=0), equal_nan=True
    )

    min_pixel_values = np.nanmin(edv1.pixel_impedance_individual_offset, axis=0)
    fill_nan_with_zero = np.nan_to_num(min_pixel_values)
    assert np.all(fill_nan_with_zero == np.zeros((32, 32)))

    assert np.array_equal(
        edv1.global_impedance, np.nansum(pixel_impedance, axis=(1, 2))
    )


def test_concatenate():
    raise NotImplementedError()


def test_slice():
    raise NotImplementedError()


def test_copy():
    raise NotImplementedError()
