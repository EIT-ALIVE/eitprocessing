import collections
import copy
import itertools
from functools import lru_cache
from importlib import resources

import yaml
from anytree import Node
from anytree.importer import DictImporter
from anytree.search import find_by_attr
from typing_extensions import Self

COMPACT_YAML_FILE_MODULE = "eitprocessing.config"
COMPACT_YAML_FILE_NAME = "categories-compact.yaml"


class Category(Node):
    """Data category indicating what type of information is saved in an object.

    Categories are nested, where more specific categories are nested inside more general categories. The root category
    is simply named 'category'. Categories have a unique name within the entire tree.

    To check the existence of a category with name <name> within a category tree, either as subcategory or
    subsub(...)category, `category.has_subcategory("<name>")` can be used. The keyword `in` can be used as a shorthand.

    Example:
    >>> "tea" in category  # is the same as:
    >>> category.has_subcategory("tea")

    To select a subcategory, category["<name>"] can be used. You can select multiple categories at once. This will
    create a new tree with a temporary root, containing only the selected categories.

    Example:
    >>> foobar = categories["foo", "bar"]
    >>> print(foobar)  # Category('/temporary root')
    >>> print(foobar.children)  # (Category('/temporary root/foo'), Category('/temporary root/bar'))

    Categories can be hand-crafted, created from a dictionary or a YAML string. See [`anytree.DictionaryImporter`
    documentation](https://anytree.readthedocs.io/en/latest/importer/dictimporter.html) for more info on the dictionary
    format. [anytree documentation on YAML import/export](https://anytree.readthedocs.io/en/latest/tricks/yaml.html)
    shows the relevant structure of a normal YAML string.

    Categories also supports a compact YAML format, where each category containing a subcategory is sequence. Categories
    without subcategories are strings in those sequences.

    ```yaml
    root:
    - sub 1 (without subcategories)
    - sub 2 (with subcategories):
      - sub a (without subcategories)
    ```

    Each type of data that is attached to an eitprocessing object should be categorized as one of the available types of
    data. This allows algorithms to check whether it can apply itself to the provided data, preventing misuse of
    algorithms.

    Example:
    >>> categories = get_default_categories()
    >>> print(categories)  # Category('/category')
    >>> print("pressure" in categories)  # True
    >>> categories["pressure"]  # Category('/category/physical measurements/pressure')
    """

    def has_subcategory(self, subcategory: str) -> bool:
        """Check whether this category contains a subcategory.

        Returns True if the category and subcategory both exist. Returns False if the category exists, but the
        subcategory does not. Raises a ValueError

        Attr:
            category: the category to be checked as an ancestor of the subcategory. This categroy should exist.
            subcategory: the subcategory to be checked as a descendent of the category.

        Returns:
            bool: whether subcategory exists as a descendent of category.

        Raises:
            ValueError: if category does not exist.
        """
        if isinstance(subcategory, type(self)):
            return subcategory is self or subcategory in self.descendants

        return bool(find_by_attr(self, subcategory, name="name"))

    def __getitem__(self, name: str | tuple[str]):
        if isinstance(name, str):
            node = find_by_attr(self, name, name="name")
            if not node:
                msg = f"Category {name} does not exist."
                raise ValueError(msg)
            return node

        temporary_root = Category(name="temporary root")
        temporary_root.children = [copy.deepcopy(self[name_]) for name_ in name]

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
                category.children = [parse_node(child_node) for child_node in node[key]]

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
