from collections.abc import Callable

import numpy as np
import pytest

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.pixelmap import PixelMap
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.roi import PixelMask
from eitprocessing.roi.pixelmaskcollection import PixelMaskCollection


@pytest.fixture
def numpy_array() -> Callable:
    return lambda: np.random.default_rng().random((32, 32))


@pytest.fixture
def pixel_map(numpy_array: Callable) -> Callable:
    return lambda: PixelMap(numpy_array(), label="test_map")


@pytest.fixture
def anonymous_boolean_mask(numpy_array: Callable) -> Callable:
    return lambda: PixelMask(numpy_array() > 0.5)


@pytest.fixture
def labelled_boolean_mask(anonymous_boolean_mask: Callable):
    return lambda label: anonymous_boolean_mask().update(label=label)


@pytest.fixture
def anonymous_float_mask(numpy_array: Callable) -> Callable:
    def factory() -> PixelMask:
        numpy_array_ = numpy_array() * 1.2 - 0.2
        numpy_array_[numpy_array_ <= 0] = np.nan
        return PixelMask(numpy_array_)

    return factory


@pytest.fixture
def labelled_float_mask(anonymous_float_mask: Callable) -> Callable:
    return lambda label: anonymous_float_mask().update(label=label)


def test_boolean_mask_factory_unique(anonymous_boolean_mask: Callable):
    mask1 = anonymous_boolean_mask()
    mask2 = anonymous_boolean_mask()
    assert not np.array_equal(mask1.mask, mask2.mask)


def test_float_mask_factory_unique(anonymous_float_mask: Callable):
    mask1 = anonymous_float_mask()
    mask2 = anonymous_float_mask()
    assert not np.array_equal(mask1.mask, mask2.mask)


def test_init_with_label(anonymous_boolean_mask: Callable):
    mask = anonymous_boolean_mask()
    collection = PixelMaskCollection([mask], label="test_collection")
    assert collection.label == "test_collection"


def test_init_without_label(anonymous_boolean_mask: Callable):
    mask = anonymous_boolean_mask()
    collection = PixelMaskCollection([mask])
    assert collection.label is None


def test_init_with_labelled_masks(labelled_boolean_mask: Callable):
    pm1 = labelled_boolean_mask("mask1")
    pm2 = labelled_boolean_mask("mask2")
    collection1 = PixelMaskCollection([pm1, pm2])
    assert collection1.masks == {"mask1": pm1, "mask2": pm2}

    collection2 = PixelMaskCollection({"mask2": pm2, "mask1": pm1})
    assert collection2 == collection1


def test_init_with_anonymous_masks(anonymous_boolean_mask: Callable):
    pm1 = anonymous_boolean_mask()
    pm2 = anonymous_boolean_mask()
    collection1 = PixelMaskCollection([pm1, pm2])
    assert collection1.masks == {0: pm1, 1: pm2}

    collection2 = PixelMaskCollection({0: pm1, 1: pm2})
    assert collection1 == collection2


def test_init_with_dict_integer_keys_unordered_raises(anonymous_boolean_mask: Callable):
    pm1 = anonymous_boolean_mask()
    pm2 = anonymous_boolean_mask()

    _ = PixelMaskCollection({1: pm1, 0: pm2})  # the order does not matter

    with pytest.raises(
        ValueError, match="Anonymous masks should be indexed with consecutive integers starting from 0."
    ):
        _ = PixelMaskCollection({1: pm1, 2: pm2})

    with pytest.raises(
        ValueError, match="Anonymous masks should be indexed with consecutive integers starting from 0."
    ):
        _ = PixelMaskCollection({0: pm1, 2: pm2})

    with pytest.raises(
        ValueError, match="Anonymous masks should be indexed with consecutive integers starting from 0."
    ):
        _ = PixelMaskCollection({0: pm1, -1: pm2})


def test_init_with_mixed_labelled_and_anonymous_masks_raises(
    labelled_boolean_mask: Callable, anonymous_boolean_mask: Callable
):
    pm1 = labelled_boolean_mask("mask1")
    pm2 = anonymous_boolean_mask()
    with pytest.raises(ValueError, match="Cannot mix labelled and anonymous masks in a collection."):
        PixelMaskCollection([pm1, pm2])


def test_init_with_dict_label_mismatch_raises(labelled_boolean_mask: Callable):
    pm1 = labelled_boolean_mask("mask1")
    pm2 = labelled_boolean_mask("mask2")
    with pytest.raises(KeyError, match="Keys should match the masks' label."):
        PixelMaskCollection({"mask1": pm1, "mask3": pm2})


def test_init_with_dict_mixed_labelled_and_anonymous_raises(
    labelled_boolean_mask: Callable, anonymous_boolean_mask: Callable
):
    pm1 = labelled_boolean_mask("mask1")
    pm2 = anonymous_boolean_mask()
    with pytest.raises(ValueError, match="Cannot mix labelled and anonymous masks in a collection."):
        PixelMaskCollection({"mask1": pm1, 0: pm2})


def test_init_with_wrong_input_raises(anonymous_boolean_mask: Callable):
    with pytest.raises(TypeError, match="Expected a list or a dictionary, got"):
        _ = PixelMaskCollection((anonymous_boolean_mask(),))

    with pytest.raises(TypeError, match="Expected a list or a dictionary, got"):
        _ = PixelMaskCollection(anonymous_boolean_mask())

    with pytest.raises(TypeError, match="Expected a list or a dictionary, got"):
        _ = PixelMaskCollection(0)


def test_init_with_non_pixelmasks_raises():
    with pytest.raises(TypeError, match="All items must be instances of PixelMask."):
        _ = PixelMaskCollection([1, 2, 3])

    with pytest.raises(TypeError, match="All items must be instances of PixelMask."):
        _ = PixelMaskCollection({"mask1": 1, "mask2": 2})


def test_is_anonymous_property_labelled(labelled_boolean_mask: Callable):
    pm = labelled_boolean_mask("mask1")
    collection = PixelMaskCollection([pm])
    assert not collection.is_anonymous


def test_is_anonymous_property_anonymous(anonymous_boolean_mask: Callable):
    pm = anonymous_boolean_mask()
    collection = PixelMaskCollection([pm])
    assert collection.is_anonymous


def test_apply_to_numpy_array_labelled(labelled_boolean_mask: Callable, numpy_array: Callable):
    pm1 = labelled_boolean_mask("mask1")
    pm2 = labelled_boolean_mask("mask2")

    data = numpy_array()

    collection = PixelMaskCollection([pm1, pm2])
    result = collection.apply(data)

    assert len(result) == 2
    assert isinstance(result, dict)
    assert all(isinstance(value, np.ndarray) for value in result.values())
    assert all(isinstance(key, str) for key in result)
    assert result.keys() == collection.masks.keys()

    assert np.array_equal(result["mask1"], pm1.apply(data), equal_nan=True)
    assert np.array_equal(result["mask2"], pm2.apply(data), equal_nan=True)


def test_apply_to_numpy_array_anonymous(anonymous_boolean_mask: Callable, numpy_array: Callable):
    pm1 = anonymous_boolean_mask()
    pm2 = anonymous_boolean_mask()

    data = numpy_array()

    collection = PixelMaskCollection([pm1, pm2])
    result = collection.apply(data)

    assert len(result) == 2
    assert isinstance(result, dict)
    assert all(isinstance(value, np.ndarray) for value in result.values())
    assert all(isinstance(key, int) for key in result)
    assert result.keys() == collection.masks.keys()

    assert np.array_equal(result[0], pm1.apply(data), equal_nan=True)
    assert np.array_equal(result[1], pm2.apply(data), equal_nan=True)


def test_apply_to_numpy_data_label_format(anonymous_boolean_mask: Callable, numpy_array: Callable):
    pm1 = anonymous_boolean_mask()
    pm2 = anonymous_boolean_mask()

    data = numpy_array()

    collection = PixelMaskCollection([pm1, pm2])
    with pytest.raises(ValueError, match=r"label_format is not applicable"):
        _ = collection.apply(data, label_format="masked_{mask_label}")


def test_apply_to_eitdata_labelled(draeger1: Sequence, labelled_boolean_mask: Callable):
    eit_data = draeger1.eit_data["raw"][:100]

    pm1 = labelled_boolean_mask("mask1")
    pm2 = labelled_boolean_mask("mask2")
    collection = PixelMaskCollection([pm1, pm2])
    result = collection.apply(eit_data)

    assert isinstance(result, dict)
    assert len(result) == 2
    assert all(isinstance(value, EITData) for value in result.values())
    assert all(isinstance(key, str) for key in result)
    assert result.keys() == collection.masks.keys()

    assert np.array_equal(result["mask1"].pixel_impedance, pm1.apply(eit_data).pixel_impedance, equal_nan=True)
    assert np.array_equal(result["mask2"].pixel_impedance, pm2.apply(eit_data).pixel_impedance, equal_nan=True)

    # Results should be the same if providing pixel_impedance array directly
    assert np.array_equal(result["mask1"].pixel_impedance, pm1.apply(eit_data.pixel_impedance), equal_nan=True)
    assert np.array_equal(result["mask2"].pixel_impedance, pm2.apply(eit_data.pixel_impedance), equal_nan=True)


def test_apply_to_pixelmap_labelled(pixel_map: Callable, labelled_boolean_mask: Callable):
    pixel_map_instance = pixel_map()

    pm1 = labelled_boolean_mask("mask1")
    pm2 = labelled_boolean_mask("mask2")
    collection = PixelMaskCollection([pm1, pm2])
    result = collection.apply(pixel_map_instance)

    assert isinstance(result, dict)
    assert len(result) == 2
    assert all(isinstance(value, PixelMap) for value in result.values())
    assert all(isinstance(key, str) for key in result)
    assert result.keys() == collection.masks.keys()

    assert np.array_equal(result["mask1"].values, pm1.apply(pixel_map_instance).values, equal_nan=True)
    assert np.array_equal(result["mask2"].values, pm2.apply(pixel_map_instance).values, equal_nan=True)

    assert np.array_equal(result["mask1"].values, pm1.apply(pixel_map_instance.values), equal_nan=True)
    assert np.array_equal(result["mask2"].values, pm2.apply(pixel_map_instance.values), equal_nan=True)


def test_apply_with_label_format(pixel_map: Callable, labelled_boolean_mask: Callable):
    pixel_map_instance = pixel_map()

    pm1 = labelled_boolean_mask("mask1")
    pm2 = labelled_boolean_mask("mask2")
    collection = PixelMaskCollection([pm1, pm2])

    result = collection.apply(pixel_map_instance, label_format="masked_{mask_label}")

    assert set(result.keys()) == {"mask1", "mask2"}
    assert result["mask1"].label == "masked_mask1"
    assert result["mask2"].label == "masked_mask2"


def test_apply_with_invalid_label_format_raises(pixel_map: Callable, labelled_boolean_mask: Callable):
    pixel_map_instance = pixel_map()

    collection = PixelMaskCollection([labelled_boolean_mask("mask1"), labelled_boolean_mask("mask2")])

    with pytest.raises(ValueError, match="Invalid label format"):
        _ = collection.apply(pixel_map_instance, label_format="masked_{}")

    with pytest.raises(ValueError, match="Invalid label format. Label format does not contain '{mask_label}'."):
        _ = collection.apply(pixel_map_instance, label_format="masked")

    with pytest.raises(ValueError, match="Invalid label format"):
        _ = collection.apply(pixel_map_instance, label_format="{mask_label} {}")

    with pytest.raises(ValueError, match="Invalid label format"):
        _ = collection.apply(pixel_map_instance, label_format="{mask_label} {something_else}")


def test_apply_with_invalid_data_type_raises(labelled_boolean_mask: Callable, draeger1: Sequence):
    collection = PixelMaskCollection([labelled_boolean_mask("mask1"), labelled_boolean_mask("mask2")])

    with pytest.raises(TypeError, match="Unsupported data type:"):
        _ = collection.apply("invalid_data")

    with pytest.raises(TypeError, match="Unsupported data type:"):
        _ = collection.apply([[1, 2]])

    with pytest.raises(TypeError, match="Unsupported data type:"):
        _ = collection.apply(draeger1)


def test_apply_with_label_keyword_raises(labelled_boolean_mask: Callable, pixel_map: Callable):
    pixel_map_instance = pixel_map()
    collection = PixelMaskCollection([labelled_boolean_mask("mask1"), labelled_boolean_mask("mask2")])

    with pytest.raises(ValueError, match=r"Cannot pass 'label' as a keyword argument to `apply\(\)`."):
        _ = collection.apply(pixel_map_instance, label="test")


def test_apply_with_extra_kwargs_on_array_raises(labelled_boolean_mask: Callable, numpy_array: Callable):
    array = numpy_array()
    collection = PixelMaskCollection([labelled_boolean_mask("mask1"), labelled_boolean_mask("mask2")])

    with pytest.raises(ValueError, match=r"Additional keyword arguments are not applicable for numpy arrays."):
        _ = collection.apply(array, sample_frequency="test")


def test_empty_collection_raises():
    with pytest.raises(ValueError, match="A PixelMaskCollection should contain at least one mask."):
        _ = PixelMaskCollection([])

    with pytest.raises(ValueError, match="A PixelMaskCollection should contain at least one mask."):
        _ = PixelMaskCollection({})

    with pytest.raises(
        TypeError, match=r"PixelMaskCollection.__init__\(\) missing 1 required positional argument: 'masks'"
    ):
        _ = PixelMaskCollection()


def test_add_labelled_mask(labelled_boolean_mask: Callable):
    collection = PixelMaskCollection([labelled_boolean_mask("existing_mask")])

    new_mask1 = labelled_boolean_mask("new_mask1")
    new_mask2 = labelled_boolean_mask("new_mask2")
    updated_collection = collection.add(new_mask1, new_mask2)

    assert len(collection.masks) == 1

    assert len(updated_collection.masks) == 3
    assert "new_mask1" in updated_collection.masks
    assert "new_mask2" in updated_collection.masks
    assert new_mask1 is updated_collection.masks["new_mask1"]
    assert new_mask2 is updated_collection.masks["new_mask2"]


def test_add_anonymous_mask(anonymous_boolean_mask: Callable):
    collection = PixelMaskCollection([anonymous_boolean_mask()])

    new_mask1 = anonymous_boolean_mask()
    new_mask2 = anonymous_boolean_mask()
    updated_collection = collection.add(new_mask1, new_mask2)

    assert len(collection.masks) == 1

    assert len(updated_collection.masks) == 3
    assert new_mask1 is updated_collection.masks[1]
    assert new_mask2 is updated_collection.masks[2]


def test_add_labelled_mask_overwrite(labelled_boolean_mask: Callable):
    existing_mask = labelled_boolean_mask("existing_mask")
    collection = PixelMaskCollection([existing_mask])
    assert collection.masks["existing_mask"] is existing_mask

    new_mask = labelled_boolean_mask("existing_mask")

    updated_collection1 = collection.add(new_mask, overwrite=True)
    assert len(collection.masks) == 1
    assert len(updated_collection1.masks) == 1
    assert updated_collection1.masks["existing_mask"] is new_mask

    updated_collection2 = collection.add(existing_mask=new_mask, overwrite=True)
    assert updated_collection2.masks["existing_mask"] is new_mask


def test_add_labelled_mask_duplicate_raises(labelled_boolean_mask: Callable):
    collection = PixelMaskCollection([labelled_boolean_mask("existing_mask")])

    new_mask = labelled_boolean_mask("existing_mask")

    with pytest.raises(KeyError, match="Cannot overwrite mask with the same key unless"):
        _ = collection.add(new_mask, overwrite=False)

    with pytest.raises(KeyError, match="Cannot overwrite mask with the same key unless"):
        _ = collection.add(new_mask)

    with pytest.raises(KeyError, match="Cannot overwrite mask with the same key unless"):
        _ = collection.add(existing_mask=new_mask)


def test_add_anonymous_mask_to_labelled_collection_raises(
    labelled_boolean_mask: Callable, anonymous_boolean_mask: Callable
):
    collection = PixelMaskCollection([labelled_boolean_mask("existing_mask")])
    new_mask = anonymous_boolean_mask()

    with pytest.raises(ValueError, match="Cannot mix labelled and anonymous masks in a collection."):
        _ = collection.add(new_mask)


def test_add_labelled_mask_to_anonymous_collection_raises(
    labelled_boolean_mask: Callable, anonymous_boolean_mask: Callable
):
    collection = PixelMaskCollection([anonymous_boolean_mask()])
    new_mask = labelled_boolean_mask("existing_mask")

    with pytest.raises(ValueError, match="Cannot mix labelled and anonymous masks in a collection."):
        _ = collection.add(new_mask)


def test_add_keyword_mask_to_anonymous_collection_raises(anonymous_boolean_mask: Callable):
    collection = PixelMaskCollection([anonymous_boolean_mask()])
    new_mask = anonymous_boolean_mask()

    with pytest.raises(ValueError, match="Cannot mix labelled and anonymous masks in a collection."):
        _ = collection.add(new_mask=new_mask)


def test_add_keyword_wrong_key_raises(labelled_boolean_mask: Callable):
    collection = PixelMaskCollection([labelled_boolean_mask("existing_mask")])
    new_mask = labelled_boolean_mask("new_mask")

    with pytest.raises(KeyError, match="Keys should match the masks' label."):
        _ = collection.add(other_key=new_mask)


def test_add_none_raises(anonymous_boolean_mask: Callable):
    collection = PixelMaskCollection([anonymous_boolean_mask()])
    with pytest.raises(ValueError, match="No masks provided to add."):
        _ = collection.add()


def test_add_anonymous_overwrite_warning(anonymous_boolean_mask: Callable):
    collection = PixelMaskCollection([anonymous_boolean_mask()])
    with pytest.warns(
        UserWarning,
        match="Cannot overwrite existing masks in an anonymous collection. All masks with be added instead.",
    ):
        _ = collection.add(anonymous_boolean_mask(), overwrite=True)


def test_update_method(anonymous_boolean_mask: Callable):
    collection = PixelMaskCollection([anonymous_boolean_mask()])
    assert collection.label is None

    updated_collection1 = collection.update(label="first")
    assert collection.label is None
    assert updated_collection1.label == "first"

    updated_collection2 = updated_collection1.update(label="second")
    assert collection.label is None
    assert updated_collection1.label == "first"
    assert updated_collection2.label == "second"


def test_combine_boolean():
    pm1 = PixelMask([[True, False], [False, True]])
    pm2 = PixelMask([[True, True], [False, False]])
    collection = PixelMaskCollection([pm1, pm2])

    summed_mask = collection.combine(method="sum", label="combined_sum")
    assert summed_mask.label == "combined_sum"
    assert np.array_equal(summed_mask.mask, np.array([[1.0, 1.0], [np.nan, 1.0]]), equal_nan=True)

    multiplied_mask = collection.combine(method="product", label="combined_product")
    assert multiplied_mask.label == "combined_product"
    assert np.array_equal(multiplied_mask.mask, np.array([[1.0, np.nan], [np.nan, np.nan]]), equal_nan=True)


def test_combine_weighted():
    pm1 = PixelMask([[0, 0.1], [0.2, 1]], suppress_zero_conversion_warning=True)
    pm2 = PixelMask([[1, 1], [0.3, 0.2]])
    collection = PixelMaskCollection([pm1, pm2])

    summed_mask = collection.combine(method="sum", label="combined_sum")
    assert summed_mask.label == "combined_sum"
    assert np.array_equal(summed_mask.mask, np.array([[1.0, 1.0], [0.5, 1.0]]), equal_nan=True)

    multiplied_mask = collection.combine(method="product", label="combined_product")
    assert multiplied_mask.label == "combined_product"
    assert np.array_equal(multiplied_mask.mask, np.array([[np.nan, 0.1], [0.06, 0.2]]), equal_nan=True)
