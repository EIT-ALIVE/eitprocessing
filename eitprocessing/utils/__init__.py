from typing import Protocol


class _CaptureFunc(Protocol):
    """Protocol for a function that captures intermediate results in a dictionary."""

    def __call__(self, key: str, value: object, append_to_list: bool = False, update_dict: bool = False) -> None: ...


def make_capture(captures: dict[str, object] | None) -> _CaptureFunc:  # noqa: C901
    """Return a helper function to capture intermediate results in a dictionary.

    The helper function sets key-value pairs in the provided `captures` dictionary. If `captures` is None, the helper
    function does nothing. If the `append_to_list` argument is True when calling the helper function, a new list is
    created at the specified key if it does not exist, and the value is appended to the new or existing list. If
    `update_dict` is True when calling the helper function, a new dictionary is created at the specified key if it does
    not exist, and the value is merged into the existing or new dictionary.

    Keys can not be overwritten. If a key already exists in the `captures` dictionary, a KeyError is raised unless
    `append_to_list` or `update_dict` is True.

    Args:
        captures: A dictionary to capture intermediate results. If None, no capturing will occur.

    Example:
    ```
    >>> capture = make_capture(captures)
    >>> capture("key", "value")  # sets captures["key"] = "value"
    >>> capture("list_key", 1, append_to_list=True)  # appends 1 to captures["list_key"]
    >>> capture("list_key", 2, append_to_list=True)  # appends 2 to captures["list_key"]
    >>> capture("dict_key", {"a": 1}, update_dict=True)  # updates captures["dict_key"] with {"a": 1}
    >>> capture("dict_key", {"b": 2}, update_dict=True)  # updates captures["dict_key"] with {"b": 2}
    >>> print(captures)
    {'key': 'value', 'list_key': [1, 2], 'dict_key': {'a': 1, 'b': 2}}
    >>> capture("key", "new_value")  # Raises a KeyError
    ```
    """

    def capture(key: str, value: object, append_to_list: bool = False, update_dict: bool = False) -> None:
        if captures is None:
            return

        if append_to_list and update_dict:
            msg = "Cannot append to list and update dict at the same time."
            raise ValueError(msg)

        if append_to_list:
            _append_to_list(key, value)
            return

        if update_dict:
            _update_dict(key, value)
            return

        if key in captures:
            msg = f"Key '{key}' already exists in captures. Use append_to_list or update_dict to modify it."
            raise KeyError(msg)

        captures[key] = value

    def _update_dict(key: str, value: object) -> None:
        if captures is None:
            return

        if not isinstance(value, dict):
            msg = f"Expected a dict for key '{key}', got {type(value)}."
            raise TypeError(msg)

        if key not in captures:
            dict_ = captures[key] = {}
        elif not isinstance((dict_ := captures[key]), dict):
            msg = f"Expected a dict for key '{key}', got {type(dict_)}."
            raise TypeError(msg)

        dict_.update(value)

    def _append_to_list(key: str, value: object) -> None:
        if captures is None:
            return

        if key not in captures:
            list_ = captures[key] = []

        elif not isinstance((list_ := captures[key]), list):
            msg = f"Expected a list for key '{key}', got {type(list_)}."
            raise TypeError(msg)

        list_.append(value)

    return capture
