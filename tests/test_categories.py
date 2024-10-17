import pytest

from eitprocessing.categories import Category, _IgnoreReadonly, get_default_categories

yaml_string = """
name: category
children:
- name: physical measurements
  children:
  - name: pressure
    children:
    - name: absolute pressure
    - name: relative pressure
    - name: pressure difference
  - name: impedance
    children:
    - name: absolute impedance
    - name: relative impedance
    - name: impedance difference
"""

categories_dict = {
    "name": "category",
    "children": [
        {
            "name": "physical measurements",
            "children": [
                {
                    "name": "pressure",
                    "children": [
                        {"name": "absolute pressure"},
                        {"name": "relative pressure"},
                        {"name": "pressure difference"},
                    ],
                },
                {
                    "name": "impedance",
                    "children": [
                        {"name": "absolute impedance"},
                        {"name": "relative impedance"},
                        {"name": "impedance difference"},
                    ],
                },
            ],
        },
    ],
}


@pytest.fixture
def categories_from_dict() -> Category:
    return Category.from_dict(categories_dict)


def test_no_empty_init():
    with pytest.raises(TypeError):
        _ = Category()


def test_load_from_dict():
    _ = Category.from_dict(categories_dict)


def test_load_from_yaml():
    _ = Category.from_yaml(yaml_string)


def test_get_item(categories_from_dict: Category):
    node = categories_from_dict["pressure difference"]
    assert isinstance(node, Category)
    assert node.name == "pressure difference"


def test_has_child(categories_from_dict: Category):
    assert categories_from_dict.has_subcategory("pressure difference")
    assert categories_from_dict["pressure"].has_subcategory("pressure difference")
    assert not categories_from_dict["pressure"].has_subcategory("impedance difference")

    with pytest.raises(ValueError):
        categories_from_dict["foo"]


def test_get_default_categories():
    categories = get_default_categories()
    assert isinstance(categories, Category)
    assert categories.has_subcategory("pressure")
    assert categories.has_subcategory("other")

    second_time = get_default_categories()
    assert second_time is categories


def test_get_multiple(categories_from_dict: Category):
    subset = categories_from_dict[["absolute pressure", "absolute impedance"]]
    assert len(subset.descendants) == 2
    assert {d.name for d in subset.descendants} == {"absolute pressure", "absolute impedance"}
    assert subset.has_subcategory("absolute pressure")
    assert categories_from_dict.has_subcategory("absolute pressure")
    assert categories_from_dict.has_subcategory("absolute impedance")
    assert subset["absolute pressure"] is not categories_from_dict["absolute pressure"]
    assert subset["absolute impedance"] is not categories_from_dict["absolute impedance"]


def test_get_multiple_non_unique(categories_from_dict: Category):
    with pytest.raises(ValueError):
        _ = categories_from_dict[["pressure", "absolute pressure"]]

    with pytest.raises(ValueError):
        _ = categories_from_dict[["absolute pressure", "pressure"]]


def test_contains(categories_from_dict: Category):
    assert "pressure" in categories_from_dict
    assert "pressure" in categories_from_dict["physical measurements"]
    assert "pressure" in categories_from_dict["pressure"]
    assert "absolute pressure" in categories_from_dict[["pressure", "impedance"]]

    assert categories_from_dict["pressure"] in categories_from_dict
    assert categories_from_dict["impedance"] not in categories_from_dict["pressure"]
    assert categories_from_dict["physical measurements"] not in categories_from_dict[["pressure", "impedance"]]


def test_readonly(categories_from_dict: Category):
    assert categories_from_dict.readonly
    assert categories_from_dict["pressure"].readonly
    with _IgnoreReadonly(categories_from_dict):
        assert not categories_from_dict.readonly
        assert categories_from_dict["pressure"].readonly

    # removing children raises RuntimeError, unless overridden for the children
    with pytest.raises(RuntimeError):
        categories_from_dict.children = []

    children = categories_from_dict.children
    with _IgnoreReadonly(children):
        categories_from_dict.children = []
        categories_from_dict.children = children

    # removing a parent raises RuntimeError, unless overridden
    with pytest.raises(RuntimeError):
        categories_from_dict["pressure"].parent = None

    pressure = categories_from_dict["pressure"]
    parent = pressure.parent
    with _IgnoreReadonly(pressure):
        pressure.parent = None
        pressure.parent = parent
