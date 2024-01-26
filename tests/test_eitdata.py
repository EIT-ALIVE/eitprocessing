from dataclasses import dataclass, field
from eitprocessing.continuous_data.continuous_data_collection import (
    ContinuousDataCollection,
)
from eitprocessing.eit_data import EITData
from eitprocessing.eit_data import NotEquivalent, NoVendorProvided, UnknownVendor
from eitprocessing.eit_data.draeger import DraegerEITData
from eitprocessing.eit_data.timpel import TimpelEITData
from eitprocessing.eit_data.sentec import SentecEITData
from eitprocessing.eit_data.vendor import Vendor
from eitprocessing.sparse_data.sparse_data_collection import SparseDataCollection
from eitprocessing.variants import Variant
from eitprocessing.variants.variant_collection import VariantCollection
from pathlib import Path
from typing_extensions import Self

import numpy as np
import os
import pytest

environment = os.environ.get(
    "EIT_PROCESSING_TEST_DATA",
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
)
data_directory = os.path.join(environment, "tests", "test_data")
draeger_file1 = os.path.join(data_directory, "Draeger_Test.bin")
draeger_file2 = os.path.join(data_directory, "Draeger_Test3.bin")
sentec_file = os.path.join(data_directory, "Sentec_Test.zri")
timpel_file = os.path.join(data_directory, "Timpel_Test.txt")
dummy_file = os.path.join(data_directory, "not_a_file.dummy")

framerate = 100
n_frames_a = 100
n_frames_b = 200
time_a = (np.arange(0, 100),)
time_a2 = (np.arange(100, 200),)
time_b = (np.arange(50, 200),)

draeger_filelength = 20740


@pytest.fixture
def EITDataSubA():
    @dataclass
    class EITDataSubA(EITData):
        def _from_path(self):
            return self

    return EITDataSubA


@pytest.fixture
def EITDataSubB():
    @dataclass
    class EITDataSubB(EITData):
        def _from_path(self):
            return self

    return EITDataSubB


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


@pytest.fixture
def variant_collection_a(mock_variant, variant_a):
    return VariantCollection(mock_variant, {"name_a": variant_a})


@pytest.fixture
def variant_collection_b(mock_variant, variant_a):
    return VariantCollection(mock_variant, {"name_a": variant_b})


@pytest.fixture
def data_a_vc_a_1(EITDataSubA, variant_collection_a):
    return EITDataSubA(
        path=[],
        vendor=Vendor.DRAEGER,
        label="A",
        nframes=n_frames_a,
        time=time_a,
        framerate=framerate,
        variants=variant_collection_a,
    )


# overlapping time with data_a_vc_a_1
@pytest.fixture
def data_a_vc_a_2(EITDataSubA, variant_collection_a):
    return EITDataSubA(
        path=[],
        vendor=Vendor.DRAEGER,
        label="A",
        nframes=n_frames_a,
        time=time_b,
        framerate=framerate,
        variants=variant_collection_a,
    )


# framerate different from a_type_1
@pytest.fixture
def data_a_vc_a_3(EITDataSubA, variant_collection_a):
    return EITDataSubA(
        path=[],
        vendor=Vendor.DRAEGER,
        label="A",
        nframes=n_frames_a,
        time=time_a2,
        framerate=framerate + 1,
        variants=variant_collection_a,
    )


# different variant from a_type_1
@pytest.fixture
def data_a_vc_b_1(EITDataSubA, variant_collection_b):
    return EITDataSubA(
        path=[],
        vendor=Vendor.DRAEGER,
        label="B",
        nframes=n_frames_b,
        time=time_a2,
        framerate=framerate,
        variants=variant_collection_b,
    )


# different eit data class
@pytest.fixture
def data_b_vc_a_1(EITDataSubB, variant_collection_a):
    return EITDataSubB(
        path=[],
        vendor=Vendor.DRAEGER,
        label="A",
        nframes=n_frames_a,
        time=time_a,
        framerate=framerate,
        variants=variant_collection_a,
    )


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


def test_ensure_vendor_invalid_vendor():
    with pytest.raises(UnknownVendor):
        EITData._ensure_vendor("unknown_vendor")

    with pytest.raises(UnknownVendor):
        EITData._ensure_vendor(123)

    with pytest.raises(NoVendorProvided):
        EITData._ensure_vendor("")

    with pytest.raises(NoVendorProvided):
        EITData._ensure_vendor(None)


def test_check_first_frame():
    # valid values

    result = EITData._check_first_frame(None)
    assert result == 0

    result = EITData._check_first_frame(5)
    assert result == 5

    # invalid values
    with pytest.raises(ValueError):
        EITData._check_first_frame(-3)

    with pytest.raises(TypeError):
        EITData._check_first_frame(3.5)

    with pytest.raises(TypeError):
        EITData._check_first_frame("abc")

    with pytest.raises(TypeError):
        EITData._check_first_frame(object())


def test_concatenate_valid(data_a_vc_a_1, EITDataSubA, variant_collection_a):
    b = EITDataSubA(
        path=[],
        vendor=Vendor.DRAEGER,
        label="B",
        nframes=n_frames_b,
        time=time_a2,
        framerate=framerate,
        variants=variant_collection_a,
    )

    result = EITData.concatenate(data_a_vc_a_1, b, label="Concatenated Data")
    # TODO: This is no more an instance of EITDataSubA, because the vendor class is automatically
    # used. Not sure if this is the intended behavior
    assert isinstance(result, EITData)

    assert result.label == "Concatenated Data"
    assert result.framerate == framerate
    assert result.nframes == n_frames_a + n_frames_b
    np.testing.assert_array_equal(result.time, np.concatenate((time_a, time_a2)))
    # no test for the variant concatenation. It depends on the specific variant


def test_concatenate_invalid(
    data_a_vc_a_1,
    data_a_vc_a_2,
    data_a_vc_a_3,
    data_a_vc_b_1,
    variant_a,
    variant_b,
    mock_variant,
    EITDataSubA,
):
    with pytest.raises(NotEquivalent):
        _ = EITData.concatenate(data_a_vc_a_1, data_a_vc_b_1, label="Concatenated Data")

    with pytest.raises(ValueError):
        _ = EITData.concatenate(data_a_vc_a_1, data_a_vc_a_2, label="Concatenated Data")

    with pytest.raises(NotEquivalent):
        _ = EITData.concatenate(data_a_vc_a_1, data_a_vc_a_3, label="Concatenated Data")


def test_check_equivalence_equal_objects(data_a_vc_a_1, data_a_vc_a_2, data_a_vc_a_3):
    # check equivalence in identical objects
    assert EITData.check_equivalence(data_a_vc_a_1, data_a_vc_a_1)

    # check equivalence in objects with the same class and variant collection, but different data
    assert EITData.check_equivalence(data_a_vc_a_1, data_a_vc_a_2)


def test_check_equivalence_different_timeframe(data_a_vc_a_1, data_a_vc_a_3):
    # different timeframes
    with pytest.raises(NotEquivalent):
        _ = EITData.check_equivalence(data_a_vc_a_1, data_a_vc_a_3, raise_=True)

    result = EITData.check_equivalence(data_a_vc_a_1, data_a_vc_a_3, raise_=False)
    assert not result


def test_check_equivalence_different_variant_collection(data_a_vc_a_1, data_a_vc_b_1):
    # different variant collections
    with pytest.raises(NotEquivalent):
        _ = EITData.check_equivalence(data_a_vc_a_1, data_a_vc_b_1, raise_=True)

    result = EITData.check_equivalence(data_a_vc_a_1, data_a_vc_b_1, raise_=False)

    assert not result


def test_check_equivalence_different_classes(data_a_vc_a_1, data_b_vc_a_1):
    # different EITData classes
    with pytest.raises(NotEquivalent):
        _ = EITData.check_equivalence(data_a_vc_a_1, data_b_vc_a_1, raise_=True)

    result = EITData.check_equivalence(data_a_vc_a_1, data_b_vc_a_1, raise_=False)

    assert not result


def test_from_path_illegal_path():
    for vendor in ["draeger", "timpel", "sentec"]:
        with pytest.raises(FileNotFoundError):
            _ = EITData.from_path(dummy_file, vendor=vendor)


def test_from_path_illegal_vendor():
    for file in [draeger_file1, timpel_file, sentec_file]:
        with pytest.raises(UnknownVendor):
            _ = EITData.from_path(path=file, vendor="some vendor")

    for vendor in ["draeger", "timpel"]:
        with pytest.raises(OSError):
            _ = EITData.from_path(path=sentec_file, vendor=vendor)

    # TODO: these tests will fail as the Sentec data reader is not yet merged
    for vendor in ["sentec", "draeger"]:
        with pytest.raises(OSError):
            _ = EITData.from_path(path=timpel_file, vendor=vendor)

    # TODO: these tests will fail as the Sentec data reader is not yet merged
    for vendor in ["timpel", "sentec"]:
        with pytest.raises(OSError):
            _ = EITData.from_path(path=draeger_file1, vendor=vendor)

    with pytest.raises(OSError):
        _ = EITData.from_path(path=[draeger_file1, timpel_file], vendor="draeger")


def test_from_path_draeger():
    draeger_data1 = EITData.from_path(path=draeger_file1, vendor="draeger")
    draeger_data2 = EITData.from_path(path=draeger_file2, vendor="draeger")

    assert isinstance(draeger_data1, DraegerEITData)
    assert len(draeger_data1.time) == draeger_filelength
    assert draeger_data1 != draeger_data2

    # Multiple files
    draeger_data_both = EITData.from_path(
        path=[draeger_file1, draeger_file2], vendor="draeger"
    )

    assert isinstance(draeger_data_both, DraegerEITData)
    assert len(draeger_data_both.time) == len(draeger_data1.time) + len(
        draeger_data2.time
    )


def test_from_path_non_eit_data():
    loaded_data = EITData.from_path(
        path=draeger_file1, vendor="draeger", return_non_eit_data=True
    )

    assert isinstance(loaded_data, tuple)
    assert isinstance(loaded_data[0], DraegerEITData)
    assert isinstance(loaded_data[1], ContinuousDataCollection)
    assert isinstance(loaded_data[2], SparseDataCollection)


def test_from_path_non_eit_data_not_present():
    # timpel returns just eit data
    with pytest.warns(UserWarning):
        loaded_data = EITData.from_path(
            path=timpel_file, vendor="timpel", return_non_eit_data=True
        )

        assert isinstance(loaded_data, tuple)
        assert isinstance(loaded_data[0], TimpelEITData)
        assert isinstance(loaded_data[1], ContinuousDataCollection)
        assert len(loaded_data[1]) == 0
        assert isinstance(loaded_data[2], SparseDataCollection)
        assert len(loaded_data[2]) == 0


def test_sliced_copy():
    draeger_data1 = EITData.from_path(
        path=draeger_file1, vendor="draeger", return_non_eit_data=False
    )

    timpel_data = EITData.from_path(
        path=timpel_file, vendor="timpel", return_non_eit_data=False
    )

    # sentec_data = EITData.from_path(
    #     path=sentec_file, vendor="sentec", return_non_eit_data=False
    # )

    start_index = 10
    end_index = 100

    draeger_start_time = draeger_data1.time[start_index]
    draeger_end_time = draeger_data1.time[end_index - 1]

    timpel_start_time = timpel_data.time[start_index]
    timpel_end_time = timpel_data.time[end_index - 1]

    # invalid indexes
    with pytest.raises(ValueError):
        _ = draeger_data1._sliced_copy(
            start_index=end_index, end_index=start_index, label="sliced draeger"
        )
    with pytest.raises(ValueError):
        _ = timpel_data._sliced_copy(
            start_index=end_index, end_index=start_index, label="sliced timpel"
        )
    # with pytest.raises(ValueError):
    #     _ = sentec_data._sliced_copy(
    #         start_index=end_index, end_index=start_index, label="sliced sentec"
    #     )

    # index out file length
    # TODO: add sentec data
    for data in [draeger_data1, timpel_data]:
        filelength = len(data.time)

        with pytest.raises(ValueError):
            _ = data._sliced_copy(
                start_index=filelength + 1,
                end_index=filelength + 2,
                label="sliced draeger",
            )

    draeger_data1_sliced = draeger_data1._sliced_copy(
        start_index=start_index, end_index=end_index, label="sliced draeger"
    )

    timpel_data_sliced = timpel_data._sliced_copy(
        start_index=start_index, end_index=end_index, label="sliced timpel"
    )
    # sentec_data_sliced = sentec_data._sliced_copy(
    #     start_index=start_index, end_index=end_index, label="sliced sentec"
    # )

    assert isinstance(draeger_data1_sliced, DraegerEITData)
    assert draeger_data1_sliced.vendor == Vendor.DRAEGER
    assert isinstance(timpel_data_sliced, TimpelEITData)
    assert timpel_data_sliced.vendor == Vendor.TIMPEL
    # assert isinstance(sentec_data_sliced, SentecEITData)
    # assert sentec_data_sliced.vendor == Vendor.SENTEC

    # TODO: add sentec data
    for data in [draeger_data1_sliced, timpel_data_sliced]:
        assert len(data.time) == end_index - start_index

        if data == draeger_data1_sliced:
            start_time = draeger_start_time
            end_time = draeger_end_time
        elif data == timpel_data_sliced:
            start_time = timpel_start_time
            end_time = timpel_end_time

        assert data.time[0] == start_time
        assert data.time[-1] == end_time
