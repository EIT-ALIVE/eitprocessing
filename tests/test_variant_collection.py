import copy
import pytest
from eitprocessing.variants import Variant
from eitprocessing.variants.variant_collection import VariantCollection
from .test_variant import VariantSubA
from .test_variant import VariantSubB


@pytest.fixture
def NonVariant():
    class NonVariant:
        pass

    return NonVariant


@pytest.fixture
def variant_A_a(VariantSubA):
    return VariantSubA("name_a", "label_a", "description_a", data=[])


@pytest.fixture
def variant_A_b(VariantSubA):
    return VariantSubA("name_b", "label_b", "description_b", data=[])


@pytest.fixture
def variant_B_a(VariantSubB):
    return VariantSubB("name_a", "label_a", "description_a", data=[])


@pytest.fixture
def variant_B_b(VariantSubB):
    return VariantSubB("name_b", "label_a", "description_b", data=[])


def test_compare(
    VariantSubA, VariantSubB, variant_A_a, variant_A_b, variant_B_a, variant_B_b
):
    vc1 = VariantCollection(VariantSubA, name_a=variant_A_a)
    vc2 = VariantCollection(VariantSubA, name_a=variant_A_b)
    assert vc1 != vc2

    vc3 = VariantCollection(VariantSubB, {"name_a": variant_B_a, "name_b": variant_B_b})
    assert vc1 != vc3

    vc4 = VariantCollection(VariantSubA)
    variant_A_a_copy = VariantSubA("name_a", "label_a", "description_a", data=[])
    vc4.add(variant_A_a_copy)

    assert variant_A_a == variant_A_a_copy
    assert variant_A_a is not variant_A_a_copy
    assert vc1 == vc4
    assert vc1 is not vc4

    vc5 = VariantCollection(Variant, name_a=variant_A_a)
    assert vc1 != vc5


def test_init(VariantSubA, variant_A_a, variant_A_b, NonVariant):
    _ = VariantCollection(Variant)
    _ = VariantCollection(VariantSubA)
    with pytest.raises(TypeError):
        _ = VariantCollection()  # type: ignore

    with pytest.raises(TypeError):
        _ = VariantCollection(NonVariant)

    # tests whether four types of initializing dicts all have the same result
    vc1 = VariantCollection(VariantSubA, {"name_a": variant_A_a, "name_b": variant_A_b})
    vc2 = VariantCollection(VariantSubA, name_a=variant_A_a, name_b=variant_A_b)
    vc3 = VariantCollection(
        VariantSubA, [("name_a", variant_A_a), ("name_b", variant_A_b)]
    )
    vc4 = VariantCollection(VariantSubA)
    vc4["name_a"] = variant_A_a
    vc4["name_b"] = variant_A_b

    assert vc1 == vc2 == vc3 == vc4


def test_set_item_add(variant_A_a, variant_A_b, variant_B_b, VariantSubA, NonVariant):
    """Tests whether the `add()` method sets items as expected"""
    vc1 = VariantCollection(Variant)
    vc2 = copy.deepcopy(vc1)

    vc1.add(variant_A_a)
    vc2["name_a"] = variant_A_a
    assert vc1 == vc2
    assert len(vc1) == len(vc2) == 1

    with pytest.raises(TypeError):
        vc1.add(NonVariant())

    vc3 = VariantCollection(VariantSubA)
    vc3.add(variant_A_a)
    with pytest.raises(TypeError):
        vc3.add(variant_B_a)

    vc4 = VariantCollection(VariantSubA)
    vc4.add(variant_A_a, variant_A_b)
    assert len(vc4) == 2
    assert "name_a" in vc4
    assert "name_b" in vc4


def test_keys(VariantSubA, variant_A_a, variant_A_b, variant_B_a, variant_B_b):
    vc1 = VariantCollection(Variant)
    vc1.add(variant_A_a)

    # don't allow overwriting existing key
    assert variant_A_a.name == variant_B_a.name
    with pytest.raises(KeyError):
        vc1.add(variant_B_a)
    with pytest.raises(KeyError):
        vc1["name_a"] = variant_B_a

    # allow overwriting existing key explictely
    vc1.add(variant_B_a, overwrite=True)

    # don't allow writing with wrong key
    with pytest.raises(KeyError):
        vc1["some_name"] = variant_B_b


def test_check_equivalence(VariantSubA):
    v1a = VariantSubA("name_a", "label_a", "description_a", data=[1, 2, 3])
    v1b = VariantSubA("name_b", "label_b", "description_b", data=[4, 5, 6])
    v1c = VariantSubA("name_c", "label_c", "description_c", data=[7, 8, 9])
    vc1 = VariantCollection(VariantSubA)
    vc1.add(v1a, v1b, v1c)

    vc3 = VariantCollection(Variant)
    vc3.add(v1a, v1b, v1c)

    assert dict(vc1) == dict(vc3)
    assert not VariantCollection.isequivalent(vc1, vc3)

    v2a = VariantSubA("name_a", "label_a", "description_a", data=[10, 11, 12])
    v2b = VariantSubA("name_b", "label_b", "description_b", data=[13, 14, 15])
    v2c = VariantSubA("name_c", "label_c", "description_c", data=[16, 17, 18])
    vc2 = VariantCollection(VariantSubA)
    vc2.add(v2a, v2b)

    assert not VariantCollection.isequivalent(vc1, vc2)
    vc2.add(v2c)
    assert VariantCollection.isequivalent(vc1, vc2)

    v1a.params = dict(key="value")
    assert not VariantCollection.isequivalent(vc1, vc2)
    v2a.params = dict(key="value")
    assert VariantCollection.isequivalent(vc1, vc2)


def test_concatenate(VariantSubA):
    v1a = VariantSubA("name_a", "label_a", "description_a", data=[1, 2, 3])
    v1b = VariantSubA("name_b", "label_b", "description_a", data=[4, 5, 6])
    vc1 = VariantCollection(Variant)
    vc1.add(v1a, v1b)

    with pytest.raises(ValueError):
        vc1.concatenate(v1a)

    v2a = VariantSubA("name_a", "label_a", "description_a", data=[7, 8, 9])
    v2b = VariantSubA("name_b", "label_b", "description_a", data=[10, 11, 12])
    vc2 = VariantCollection(Variant)
    vc2.add(v2a)

    with pytest.raises(ValueError):
        _ = vc1.concatenate(vc2)

    vc2.add(v2b)

    vc_concat1 = vc1.concatenate(vc2)
    vc_concat2 = VariantCollection.concatenate(vc1, vc2)

    assert vc_concat1 == vc_concat2
    assert vc_concat1["name_a"].data == v1a.data + v2a.data
    assert vc_concat1["name_b"].data == v1b.data + v2b.data
    assert vc_concat1["name_a"].check_equivalence(v1a)
    assert vc_concat1["name_a"].check_equivalence(v2a)
