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
    return VariantSubA("label_a", "description_a", data=[])


@pytest.fixture
def variant_A_b(VariantSubA):
    return VariantSubA("label_b", "description_b", data=[])


@pytest.fixture
def variant_B_a(VariantSubA):
    return VariantSubA("label_a", "description_a", data=[])


@pytest.fixture
def variant_B_b(VariantSubA):
    return VariantSubA("label_b", "description_b", data=[])


def test_compare(
    VariantSubA, VariantSubB, variant_A_a, variant_A_b, variant_B_a, variant_B_b
):
    vc1 = VariantCollection(VariantSubA, label_a=variant_A_a)
    vc2 = VariantCollection(VariantSubA, label_a=variant_A_b)
    assert vc1 != vc2

    vc3 = VariantCollection(
        VariantSubB, {"label_a": variant_B_a, "label_b": variant_B_b}
    )
    assert vc1 != vc3

    vc4 = VariantCollection(VariantSubA)
    variant_A_a_copy = VariantSubA("label_a", "description_a", data=[])
    vc4.add(variant_A_a_copy)

    assert variant_A_a == variant_A_a_copy
    assert variant_A_a is not variant_A_a_copy
    assert vc1 == vc4
    assert vc1 is not vc4

    vc5 = VariantCollection(Variant, label_a=variant_A_a)
    assert vc1 != vc5


def test_init(VariantSubA, variant_A_a, variant_A_b, NonVariant):
    _ = VariantCollection(Variant)
    _ = VariantCollection(VariantSubA)
    with pytest.raises(TypeError):
        _ = VariantCollection()

    with pytest.raises(TypeError):
        _ = VariantCollection(NonVariant)

    # tests whether four types of initializing dicts all have the same result
    vc1 = VariantCollection(
        VariantSubA, {"label_a": variant_A_a, "label_b": variant_A_b}
    )
    vc2 = VariantCollection(VariantSubA, label_a=variant_A_a, label_b=variant_A_b)
    vc3 = VariantCollection(
        VariantSubA, [("label_a", variant_A_a), ("label_b", variant_A_b)]
    )
    vc4 = VariantCollection(VariantSubA)
    vc4["label_a"] = variant_A_a
    vc4["label_b"] = variant_A_b

    assert vc1 == vc2 == vc3 == vc4


def test_set_item_add(variant_A_a, variant_A_b, variant_B_b, VariantSubA, NonVariant):
    """Tests whether the `add()` method sets items as expected"""
    vc1 = VariantCollection(Variant)
    vc2 = copy.deepcopy(vc1)

    vc1.add(variant_A_a)
    vc2["label_a"] = variant_A_a
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
    assert "label_a" in vc4
    assert "label_b" in vc4


def test_keys(VariantSubA, variant_A_a, variant_A_b, variant_B_a, variant_B_b):
    vc1 = VariantCollection(Variant)
    vc1.add(variant_A_a)

    # don't allow overwriting existing key
    assert variant_A_a.label == variant_B_a.label
    with pytest.raises(KeyError):
        vc1.add(variant_B_a)
    with pytest.raises(KeyError):
        vc1["label_a"] = variant_B_a

    # allow overwriting existing key explictely
    vc1.add(variant_B_a, overwrite=True)

    # don't allow writing with wrong key
    with pytest.raises(KeyError):
        vc1["some_label"] = variant_B_b


def test_check_equivalence():
    ...


def test_concatenate():
    ...
