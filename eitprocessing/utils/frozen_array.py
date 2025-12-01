from typing import Literal

import numpy as np

DEFAULT_FREEZE_METHOD = "memoryview"


def freeze_array(a: np.ndarray, *, method: Literal["flag", "memoryview"] = DEFAULT_FREEZE_METHOD) -> np.ndarray:
    """Return a read-only array that cannot be made writeable again."""
    match method:
        case "flag":
            if a.flags["WRITEABLE"]:
                a = a.copy()
                a.flags["WRITEABLE"] = False
            return a  # is already read-only, e.g., a view of a read-only array
        case "memoryview":
            a_c = np.ascontiguousarray(a)
            ro_buf = memoryview(a_c).toreadonly()
            return np.frombuffer(ro_buf, dtype=a_c.dtype).reshape(a_c.shape)
        case _:
            msg = f"Invalid method: {method!r}"
            raise ValueError(msg)
