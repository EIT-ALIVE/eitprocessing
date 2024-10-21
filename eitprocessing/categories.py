import collections
import copy
import itertools
from collections.abc import Sequence
from functools import lru_cache
from importlib import resources

import yaml
from anytree import Node
from anytree.importer import DictImporter
from anytree.search import find_by_attr
from typing_extensions import Self

from eitprocessing.datahandling import DataContainer

COMPACT_YAML_FILE_MODULE = "eitprocessing.config"
COMPACT_YAML_FILE_NAME = "categories-compact.yaml"


class Category(Node):
    """Data category indicating what type of information is saved in an object.

    Categories are nested, where more specific categories are nested inside more general categories. The root category
    is simply named 'category'. Categories have a unique name within the entire tree.

    To check the existence of a category with name <name> within a category tree, either as subcategory or
    subsub(...)category, `category.has_subcategory("<name>")` can be used. The keyword `in` can be used as a shorthand.

    Example:
    ```
    >>> "tea" in category  # is the same as:
    True
    >>> category.has_subcategory("tea")
    True
    ```

    To select a subcategory, `category["<name>"]` can be used. You can select multiple categories at once. This will
    create a new tree with a temporary root, containing only the selected categories.

    Example:
    ```
    >>> foobar = categories["foo", "bar"]
    >>> print(foobar)
    Category('/temporary root')
    >>> print(foobar.children)
    (Category('/temporary root/foo'), Category('/temporary root/bar'))
    ```

    Categories can be hand-crafted, created from a dictionary or a YAML string. See [`anytree.DictionaryImporter`
    documentation](https://anytree.readthedocs.io/en/latest/importer/dictimporter.html) for more info on the dictionary
    format. [anytree documentation on YAML import/export](https://anytree.readthedocs.io/en/latest/tricks/yaml.html)
    shows the relevant structure of a normal YAML string.

    Categories also supports a compact YAML format, where each category containing a subcategory is a sequence.
    Categories without subcategories are strings in those sequences.

    ```yaml
    root:
    - sub 1 (without subcategories)
    - sub 2 (with subcategories):
      - sub a (without subcategories)
    ```

    Categories are read-only by default, as they should not be edited by the end-user during runtime. Consider editing
    the config file instead.

    Each type of data that is attached to an eitprocessing object should be categorized as one of the available types of
    data. This allows algorithms to check whether it can apply itself to the provided data, preventing misuse of
    algorithms.

    Example:
    ```
    >>> categories = get_default_categories()
    >>> print(categories)
    Category('/category')
    >>> print("pressure" in categories)
    True
    >>> categories["pressure"]
    Category('/category/physical measurement/pressure')
    ```
    """

    readonly = True

    def has_subcategory(self, subcategory: str) -> bool:
        """Check whether this category contains a subcategory.

        Returns True if the category and subcategory both exist. Returns False if the category exists, but the
        subcategory does not. Raises a ValueError

        Attr:
            category: the category to be checked as an ancestor of the subcategory. This category should exist.
            subcategory: the subcategory to be checked as a descendent of the category.

        Returns:
            bool: whether subcategory exists as a descendent of category.

        Raises:
            ValueError: if category does not exist.
        """
        if isinstance(subcategory, type(self)):
            return subcategory is self or subcategory in self.descendants

        return bool(find_by_attr(self, subcategory, name="name"))

    def __init__(self, name: str, parent: Self | None = None) -> None:
        super().__init__(name=name)
        with _IgnoreReadonly(self):
            self.parent = parent

    def __getitem__(self, name: str | tuple[str]):
        if isinstance(name, str):
            node = find_by_attr(self, name, name="name")
            if not node:
                msg = f"Category {name} does not exist."
                raise ValueError(msg)
            return node

        temporary_root = Category(name="temporary root")

        child_categories = [copy.deepcopy(self[name_]) for name_ in name]

        with _IgnoreReadonly(child_categories):
            temporary_root.children = child_categories

        return temporary_root

    def __contains__(self, item: str | Self):
        return self.has_subcategory(item)

    @classmethod
    def from_yaml(cls, string: str) -> Self:
        """Load categories from YAML file."""
        dict_ = yaml.load(string, Loader=yaml.SafeLoader)
        return cls.from_dict(dict_)

    @classmethod
    def from_compact_yaml(cls, string: str) -> Self:
        """Load categories from compact YAML file."""

        def parse_node(node: str | dict) -> Category:
            if isinstance(node, str):
                return Category(name=node)

            if isinstance(node, dict):
                if len(node) > 1:
                    msg = "Category data is malformed."
                    raise ValueError(msg)
                key = next(iter(node.keys()))
                category = Category(name=key)
                child_categories = [parse_node(child_node) for child_node in node[key]]

                with _IgnoreReadonly(child_categories):
                    category.children = child_categories

                return category

            msg = f"Supplied node should be str or dict, not {type(node)}."
            raise TypeError(msg)

        data = yaml.load(string, Loader=yaml.SafeLoader)
        return parse_node(data)

    @classmethod
    def from_dict(cls, dictionary: dict) -> Self:
        """Create categories from dictionary."""
        return DictImporter(nodecls=Category).import_(dictionary)

    def _pre_attach_children(self, children: list[Self]) -> None:
        """Checks for non-unique categories before adding them to an existing category tree."""
        for child in children:
            for node in [child, *child.descendants]:
                # Checks whether the names of children to be added and their descendents don't already exist in the
                # tree.
                if node.name in self.root:
                    msg = f"Can't add non-unique category {node.name}"
                    raise ValueError(msg)

        for child_a, child_b in itertools.permutations(children, 2):
            for child in [child_a, *child_a.descendants]:
                if child.name in child_b:
                    # Checks whether any child or their descendents exist in other children to be added
                    msg = f"Can't add non-unique category '{child.name}'"
                    raise ValueError(msg)

    def _pre_attach(self, parent: Self) -> None:  # noqa: ARG002
        if self.readonly:
            msg = "Can't attach read-only Category to another Category."
            raise RuntimeError(msg)

    def _pre_detach(self, parent: Self) -> None:  # noqa: ARG002
        if self.readonly:
            msg = "Can't detach read-only Category from another Category."
            raise RuntimeError(msg)

    def _check_unique(self, raise_: bool = False) -> bool:
        names = [self.name, *(node.name for node in self.descendants)]

        if len(names) == len(set(names)):
            return True

        if not raise_:
            return False

        count = collections.Counter(names)
        non_unique_names = [name for name, count in count.items() if count > 1]
        joined_names = ", ".join(f"'{name}'" for name in non_unique_names)
        msg = f"Some nodes have non-unique names: {joined_names}."
        raise ValueError(msg)


@lru_cache
def get_default_categories() -> Category:
    """Loads the default categories from file.

    This returns the categories used in the eitprocessing package. The root category is simply called 'root'. All other
    categories are subdivided into physical measurements, calculated values and others.

    This function is cached, meaning it only loads the data once, and returns the same object every time afterwards.
    """
    yaml_file_path = resources.files(COMPACT_YAML_FILE_MODULE).joinpath(COMPACT_YAML_FILE_NAME)
    with yaml_file_path.open("r") as fh:
        return Category.from_compact_yaml(fh.read())


def check_category(data: DataContainer, category: str, *, raise_: bool = False) -> bool:
    """Check whether the category of a dataset is a given category or one of it's subcategories.

    Example:
    >>> data = ContinuousData(..., category="impedance", ...)
    >>> check_category(data, "impedance")  # True
    >>> check_category(data, "pressure")  # False
    >>> check_category(data, "pressure", raise_=True)  # raises ValueError
    >>> check_category(data, "does not exist", raise_=False)  # raises ValueError

    Args:
        data: DataContainer object with a `category` attribute.
        category: Category to match the data category against. The data category will match this and all subcategories.
        raise_: Keyword only. Whether to raise an exception if the data is not a (sub)category.

    Returns:
        bool: Whether the data category matches.

    Raises:
        ValueError: If the provided category does not exist.
        ValueError: If the data category does not match the provided category.
    """
    categories = get_default_categories()

    if category not in categories:
        msg = f"Category '{category}' does not exist in the default categories."
        raise ValueError(msg)

    if data.category in categories[category]:
        return True

    if raise_:
        msg = f"`This method will only work on '{category}' data, not '{data.category}'."
        raise ValueError(msg)

    return False


class _IgnoreReadonly:
    """Context manager allowing temporarily ignoring the read-only attribute.

    For internal use only.

    Example:
    >>> foo = categories["foo"]
    >>> foo.parent = None  # raises RuntimeError
    >>> with _IgnoreReadonly(foo):
    >>>    foo.parent = None  # does not raise RuntimeError
    """

    items: Sequence[Category]

    def __init__(self, items: Category | Sequence[Category]):
        if not isinstance(items, Sequence):
            items = (items,)

        self.items = items

    def __enter__(self) -> None:
        for item in self.items:
            item.readonly = False

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        for item in self.items:
            item.readonly = True
