from dataclasses import dataclass
from dataclasses import field
from typing import get_type_hints
from unittest.mock import patch
import pytest
from typing_extensions import Self
from eitprocessing.helper import NotEquivalent
from eitprocessing.variants import Variant


@pytest.fixture
def variant_a(VariantSubA, make_params):
    return VariantSubA(
        "name_a", "label_a", "description_a", params=make_params(), data=[1, 2, 3]
    )


@pytest.fixture
def variant_a_copy(VariantSubA, make_params):
    return VariantSubA(
        "name_a", "label_a", "description_a", params=make_params(), data=[1, 2, 3]
    )


@pytest.fixture
def variant_b(VariantSubA, make_params):
    return VariantSubA(
        "name_b", "label_b", "description_b", make_params(), data=[4, 5, 6]
    )


@pytest.fixture
def variant_c(VariantSubB, make_params):
    return VariantSubB(
        "name_b", "label_b", "description_b", make_params(), data=[4, 5, 6]
    )


@pytest.fixture
def make_params():
    def _make_params():
        return dict(
            some_string="string", some_dict=dict(some_int=5, some_string="string 2")
        )

    return _make_params


@pytest.fixture
def VariantSubA():
    @dataclass
    class VariantSubA(Variant):
        data: list = field(repr=False, kw_only=True)

        def concatenate(self: Self, other: Self) -> Self:
            self.check_equivalence(other)
            return self.__class__(
                self.name,
                self.label,
                self.description,
                self.params,
                data=self.data + other.data,
            )

    return VariantSubA


@pytest.fixture
def VariantSubB():
    @dataclass
    class VariantSubB(Variant):
        data: list = field(repr=False, kw_only=True)

        def concatenate(self: Self, other: Self) -> Self:
            self.check_equivalence(other)
            return self.__class__(
                self.name,
                self.label,
                self.description,
                self.params,
                data=self.data + other.data,
            )

    return VariantSubB


def test_init(VariantSubA, make_params):
    _ = VariantSubA("name", "label", "description", data=[])
    _ = VariantSubA("name", "label", "description", make_params(), data=[])
    _ = VariantSubA("name", "label", "description", data=[], params=make_params())

    with pytest.raises(TypeError):
        # you should not be able to initialize the abstract base class Variant
        _ = Variant("name", "label", "description")

    with pytest.raises(TypeError):
        _ = VariantSubA()

    with pytest.raises(TypeError):
        _ = VariantSubA("name")

    with pytest.raises(TypeError):
        _ = VariantSubA(name="name")

    with pytest.raises(TypeError):
        _ = VariantSubA(name="name", description="description")

    with pytest.raises(TypeError):
        _ = VariantSubA("name", "label", "description")

    assert get_type_hints(VariantSubA) == dict(
        name=str, label=str, description=str, params=dict, data=list
    )

    with pytest.raises(TypeError):
        _ = VariantSubA(1, "label", "description", data=[])

    with pytest.raises(TypeError):
        _ = VariantSubA("name", 1, "description", data=[])

    with pytest.raises(TypeError):
        _ = VariantSubA("name", "label", 1, data=[])

    with pytest.raises(TypeError):
        _ = VariantSubA("name", "label", "description", data={1})


def test_equivalence(variant_a, variant_a_copy, variant_b, variant_c):
    assert variant_a is not variant_a_copy  # objects are not the same
    assert variant_a == variant_a_copy  # objects contain same attribute values
    assert variant_a != variant_b  # objects have different attribute values

    variant_a_copy.data = [4, 5, 6]
    assert variant_a != variant_a_copy  # objects contain different data
    # objects are still equivalent
    assert Variant.check_equivalence(variant_a, variant_a_copy)
    assert variant_a.check_equivalence(variant_a_copy)

    # objects with different attributes are not equivalent
    variant_a_copy.label = "different label"
    assert not Variant.check_equivalence(variant_a, variant_a_copy)
    assert not variant_a.check_equivalence(variant_a_copy)
    variant_a_copy.label = variant_a.label

    # objects with different attributes are not equivalent
    variant_a_copy.description = "different description"
    assert not Variant.check_equivalence(variant_a, variant_a_copy)
    assert not variant_a.check_equivalence(variant_a_copy)
    variant_a_copy.description = variant_a.description

    # objects with different parameters are not equivalent
    variant_a_copy.params["some_dict"]["some_string"] = "another string"
    assert not Variant.check_equivalence(variant_a, variant_a_copy)
    assert not variant_a.check_equivalence(variant_a_copy)
    variant_a_copy.params["some_dict"]["some_string"] = variant_a.params["some_dict"][
        "some_string"
    ]

    # objects with different attribute values are not equivalent
    assert not Variant.check_equivalence(variant_a, variant_b)
    assert not variant_a.check_equivalence(variant_b)
    with pytest.raises(NotEquivalent):
        assert not Variant.check_equivalence(variant_a, variant_b, raise_=True)
    with pytest.raises(NotEquivalent):
        assert not variant_a.check_equivalence(variant_b, raise_=True)

    # objects with different classes are not equivalent
    assert not Variant.check_equivalence(variant_b, variant_c)
    assert not variant_b.check_equivalence(variant_c)
