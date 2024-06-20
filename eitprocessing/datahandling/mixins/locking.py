from dataclasses import dataclass, fields

import numpy as np


@dataclass
class Lockable:
    """Adds locking functionality to subclass.

    This class provides the methods lock(), unlock(), lock_all(), unlock_all(), islocked() and islockable(), and the
    property _lock_action_defaults.
    """

    def lock(self, *attr: str) -> None:
        """Lock attributes, essentially rendering them read-only.

        Locked attributes cannot be overwritten. Attributes can be unlocked using `unlock()`.

        Args:
            *attr: any number of attributes can be passed here, all of which will be locked. Defaults to "values".

        Examples:
            >>> # lock the `values` attribute of `data`
            >>> data.lock()
            >>> data.values = [1, 2, 3] # will result in an AttributeError
            >>> data.values[0] = 1      # will result in a RuntimeError
        """
        if not len(attr):
            # default values are not allowed when using *attr, so set a default here if none is supplied
            attr = tuple(self._lock_action_defaults)
        for attr_ in attr:
            if not self.islockable(attr_):
                msg = f"Attribute {attr_} is not lockable."
                raise ValueError(msg)
            getattr(self, attr_).flags["WRITEABLE"] = False

    def unlock(self, *attr: str) -> None:
        """Unlock attributes, rendering them editable.

        Locked attributes cannot be overwritten, but can be unlocked with this function to make them editable.

        Args:
            *attr: any number of attributes can be passed here, all of which will be unlocked. Defaults to "values".

        Examples:
            >>> # lock the `values` attribute of `data`
            >>> data.lock()
            >>> data.values = [1, 2, 3] # will result in an AttributeError
            >>> data.values[0] = 1      # will result in a RuntimeError
            >>> data.unlock()
            >>> data.values = [1, 2, 3]
            >>> print(data.values)
            [1,2,3]
            >>> data.values[0] = 1      # will result in a RuntimeError
            >>> print(data.values)
            1
        """
        if not len(attr):
            # default values are not allowed when using *attr, so set a default here if none is supplied
            attr = tuple(self._lock_action_defaults)
        for attr_ in attr:
            if not self.islockable(attr_):
                msg = f"Attribute {attr_} is not (un)lockable."
                raise ValueError(msg)
            getattr(self, attr_).flags["WRITEABLE"] = True

    def lock_all(self) -> None:
        """Lock all lockable attributes.

        See lock().
        """
        for attr in vars(self):
            if self.islockable(attr):
                self.lock(attr)

    def unlock_all(self) -> None:
        """Unlock all (un)lockable attributes.

        See unlock().
        """
        for attr in vars(self):
            if self.islockable(attr):
                self.unlock(attr)

    def islocked(self, attr: str = "values") -> bool:
        """Return whether an attribute is locked.

        See lock().
        """
        return not getattr(self, attr).flags["WRITEABLE"]

    def islockable(self, attr: str = "values") -> bool:
        """Return whether an attribute is lockable.

        See lock().
        """
        return isinstance(getattr(self, attr), np.ndarray)

    @property
    def _lock_action_defaults(self) -> list[str]:
        """Returns a list of attributes that (un)locked if no arguments are provided to lock() or unlock()."""
        return [
            field.name
            for field in filter(
                lambda x: x.metadata.get("lock_action_default", False),
                fields(self),
            )
        ]
