from dataclasses import dataclass
from dataclasses import field
import numpy as np
import pytest
from typing_extensions import Self
from eitprocessing.eit_data.eit_data_variant import EITDataVariant
from eitprocessing.helper import NotEquivalent
from eitprocessing.mixins import SelectByIndex
from eitprocessing.variants import Variant


@pytest.fixture
def gen_pixel_impedance():
    def _pixel_impedance(n, baseline, amplitude):
        rng = np.random.default_rng()
        return rng.random((n, 32, 32), np.float_) * amplitude + baseline

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
    _ = EITDataVariant("label", "desc", pixel_impedance=np.zeros((100, 32, 32)))
    with pytest.raises(TypeError):
        _ = EITDataVariant("label", "desc")

    with pytest.raises(ValueError):
        _ = EITDataVariant("label", "desc", pixel_impedance=np.zeros((0, 32, 32)))

    with pytest.raises(ValueError):
        _ = EITDataVariant("label", "desc", pixel_impedance=np.zeros((32, 32, 100)))

    with pytest.raises(ValueError):
        _ = EITDataVariant("label", "desc", pixel_impedance=np.zeros((100, 32, 16)))


def test_len(gen_pixel_impedance):
    for n in range(10, 100, 10):
        pixel_impedance = gen_pixel_impedance(n, 1, 1)
        edv = EITDataVariant("label", "description", pixel_impedance=pixel_impedance)
        assert len(edv) == pixel_impedance.shape[0]


def test_eq(gen_pixel_impedance, NonEITDataVariant):
    pixel_impedance = gen_pixel_impedance(1000, 0, 100)

    edv1 = EITDataVariant(
        "label",
        "description",
        {"foo": "bar", "other": 1},
        pixel_impedance=pixel_impedance,
    )
    edv2 = EITDataVariant(
        "label",
        "description",
        {"other": 1, "foo": "bar"},
        pixel_impedance=np.copy(pixel_impedance),
    )
    nedv = NonEITDataVariant(
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
        "label",
        "description",
        {"foo": "bar", "other": 1},
        pixel_impedance=gen_pixel_impedance(1, 0, 100),
    )
    edv2 = EITDataVariant(
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


def test_properties():
    raise NotImplementedError()


def test_concatenate():
    raise NotImplementedError()


def test_slice():
    raise NotImplementedError()


def test_copy():
    raise NotImplementedError()
