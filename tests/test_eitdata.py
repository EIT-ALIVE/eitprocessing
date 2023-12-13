from dataclasses import dataclass, field
from eitprocessing.eit_data import EITData
from eitprocessing.eit_data import NotEquivalent, NoVendorProvided, UnknownVendor
from eitprocessing.eit_data.draeger import DraegerEITData
from eitprocessing.eit_data.timpel import TimpelEITData
from eitprocessing.eit_data.sentec import SentecEITData
from eitprocessing.eit_data.vendor import Vendor
from eitprocessing.variants import Variant
from eitprocessing.variants.variant_collection import VariantCollection
from pathlib import Path
from typing_extensions import Self

import numpy as np
import pytest


@pytest.fixture
def EITDataSubA():
    @dataclass
    class EITDataSubA(EITData):
        def _from_path(self):
            return self

    return EITDataSubA


@pytest.fixture
def mock_variant():
    @dataclass
    class MockVariant(Variant):
        data: list = field(repr=False, kw_only=True)

        def concatenate(self: Self, other: Self) -> Self:
            return self

    return MockVariant


@pytest.fixture
def variant_a(mock_variant):
    return mock_variant("name_a", "label_a", "description_a", data=np.arange(0, 100))


@pytest.fixture
def variant_b(mock_variant):
    return mock_variant("name_b", "label_b", "description_b", data=np.arange(50, 150))


def test_init(EITDataSubA):
    with pytest.raises(TypeError):
        # you should not be able to initialize the abstract base class EITData
        _ = EITData()

    with pytest.raises(TypeError):
        # you should not be able to initialize the class without arguments
        _ = EITDataSubA()

    _ = EITDataSubA(
        path="path", nframes=10, time=[1, 2, 3], framerate=20, vendor=Vendor.DRAEGER
    )


def test_ensure_path_list_single():
    path_string = "some/path/file.txt"
    path_obj = Path(path_string)

    result_string = EITData._ensure_path_list(path_string)
    result_obj = EITData._ensure_path_list(path_obj)

    assert result_string == [path_obj]
    assert result_obj == [path_obj]


def test_ensure_path_list_multiple():
    paths_string = ["path1/file1.txt", "path2/file2.txt", "path3/file3.txt"]
    paths_obj = [Path(p) for p in paths_string]

    result_string = EITData._ensure_path_list(paths_string)
    result_obj = EITData._ensure_path_list(paths_obj)

    assert result_string == paths_obj
    assert result_obj == paths_obj


def test_get_vendor_class_draeger():
    vendor = Vendor.DRAEGER
    result = EITData._get_vendor_class(vendor)
    assert result == DraegerEITData


def test_get_vendor_class_sentec():
    vendor = Vendor.SENTEC
    result = EITData._get_vendor_class(vendor)
    assert result == SentecEITData


def test_get_vendor_class_timpel():
    vendor = Vendor.TIMPEL
    result = EITData._get_vendor_class(vendor)
    assert result == TimpelEITData


def test_get_vendor_class_invalid_vendor():
    with pytest.raises(KeyError):
        EITData._get_vendor_class("unexistent_key")


def test_ensure_vendor_valid_draeger():
    vendor_enum = Vendor.DRAEGER
    vendor_string = "draeger"

    result_enum = EITData._ensure_vendor(vendor_enum)
    result_string = EITData._ensure_vendor(vendor_string)

    assert result_enum == vendor_enum
    assert result_string == vendor_enum


def test_ensure_vendor_valid_sentec():
    vendor_enum = Vendor.SENTEC
    vendor_string = "sentec"

    result_enum = EITData._ensure_vendor(vendor_enum)
    result_string = EITData._ensure_vendor(vendor_string)

    assert result_enum == vendor_enum
    assert result_string == vendor_enum


def test_ensure_vendor_valid_timpel():
    vendor_enum = Vendor.TIMPEL
    vendor_string = "timpel"

    result_enum = EITData._ensure_vendor(vendor_enum)
    result_string = EITData._ensure_vendor(vendor_string)

    assert result_enum == vendor_enum
    assert result_string == vendor_enum


def test_ensure_vendor_invalid_vendor_string():
    with pytest.raises(UnknownVendor):
        EITData._ensure_vendor("unknown_vendor")

    with pytest.raises(NoVendorProvided):
        EITData._ensure_vendor("")

    with pytest.raises(NoVendorProvided):
        EITData._ensure_vendor(None)


# TODO: add also other types?
# def test_ensure_vendor_invalid_vendor_type():
#     with pytest.raises(TypeError, match="must be a str or Vendor enum"):
#         EITData._ensure_vendor(123)


def test_check_first_frame_none():
    result = EITData._check_first_frame(None)
    assert result == 0


def test_check_first_frame_positive_integer():
    result = EITData._check_first_frame(5)
    assert result == 5


def test_check_first_frame_negative_integer():
    with pytest.raises(ValueError):
        EITData._check_first_frame(-3)


def test_check_first_frame_float():
    with pytest.raises(TypeError):
        EITData._check_first_frame(3.5)


# TODO: check if we want to take into account other types (to except)
# def test_check_first_frame_string():
#     with pytest.raises(TypeError):
#         EITData._check_first_frame("abc")
#
#
# def test_check_first_frame_object():
#     with pytest.raises(TypeError):
#         EITData._check_first_frame(object())


def test_concatenate_valid(variant_a, mock_variant, EITDataSubA):
    framerate = 100
    n_frames_a = 100
    n_frames_b = 200
    time_a = (np.arange(0, 100),)
    time_b = (np.arange(100, 200),)
    vc_a = VariantCollection(mock_variant, {"name_a": variant_a})
    a = EITDataSubA(
        path=[],
        vendor=Vendor.DRAEGER,
        label="A",
        nframes=n_frames_a,
        time=time_a,
        framerate=framerate,
        variants=vc_a,
    )
    b = EITDataSubA(
        path=[],
        vendor=Vendor.DRAEGER,
        label="B",
        nframes=n_frames_b,
        time=time_b,
        framerate=framerate,
        variants=vc_a,
    )

    result = EITDataSubA.concatenate(a, b, label="Concatenated Data")
    # TODO: This is no more an instance of EITDataSubA, because the vendor class is automatically
    # used. Not sure if this is the intended behavior
    assert isinstance(result, EITData)

    assert result.label == "Concatenated Data"
    assert result.framerate == framerate
    assert result.nframes == n_frames_a + n_frames_b
    np.testing.assert_array_equal(result.time, np.concatenate((time_a, time_b)))
    # no test for the variant concatenation. It depends on the specific variant


def test_concatenate_invalid(variant_a, variant_b, mock_variant, EITDataSubA):
    framerate = 100
    n_frames_a = 100
    n_frames_b = 200
    time_a = (np.arange(0, 100),)
    time_a2 = (np.arange(100, 200),)
    time_b = (np.arange(50, 200),)

    vc_a = VariantCollection(mock_variant, {"name_a": variant_a})
    vc_b = VariantCollection(mock_variant, {"name_b": variant_b})

    a_type_1 = EITDataSubA(
        path=[],
        vendor=Vendor.DRAEGER,
        label="A",
        nframes=n_frames_a,
        time=time_a,
        framerate=framerate,
        variants=vc_a,
    )

    # overlapping time with a_type_1
    a_type_2 = EITDataSubA(
        path=[],
        vendor=Vendor.DRAEGER,
        label="A",
        nframes=n_frames_a,
        time=time_b,
        framerate=framerate,
        variants=vc_a,
    )

    # framerate different from a_type_1
    a_type_3 = EITDataSubA(
        path=[],
        vendor=Vendor.DRAEGER,
        label="A",
        nframes=n_frames_a,
        time=time_a2,
        framerate=framerate + 1,
        variants=vc_a,
    )

    # Vendor different from a_type_1
    a_type_4 = EITDataSubA(
        path=[],
        vendor=Vendor.SENTEC,
        label="A",
        nframes=n_frames_a,
        time=time_a2,
        framerate=framerate,
        variants=vc_a,
    )

    # different variant from a_type_1
    b_type_1 = EITDataSubA(
        path=[],
        vendor=Vendor.DRAEGER,
        label="B",
        nframes=n_frames_b,
        time=time_a2,
        framerate=framerate,
        variants=vc_b,
    )

    with pytest.raises(NotEquivalent):
        _ = EITDataSubA.concatenate(a_type_1, b_type_1, label="Concatenated Data")

    with pytest.raises(ValueError):
        _ = EITDataSubA.concatenate(a_type_1, a_type_2, label="Concatenated Data")

    with pytest.raises(NotEquivalent):
        _ = EITDataSubA.concatenate(a_type_1, a_type_3, label="Concatenated Data")
