import warnings
from typing import Literal

import numpy as np

DEFAULT_FREEZE_METHOD = "memoryview"


def freeze_array(a: np.ndarray, *, method: Literal["flag", "memoryview"] = DEFAULT_FREEZE_METHOD) -> np.ndarray:
    """Return a read-only array that cannot be made writeable again."""
    # Memory buffers cannot represent object/structured fields safely.
    if method == "memoryview":
        dt = a.dtype
        if dt.hasobject:
            warnings.warn(
                "Cannot use 'memoryview' method for object or structured dtypes; falling back to 'flag' method.",
                RuntimeWarning,
                stacklevel=2,
            )
            method = "flag"

    match method:
        case "flag":
            # Make a copy if needed and mark it readonly (can be flipped back by a user).
            if a.flags["WRITEABLE"]:
                a = a.copy()
            a.flags["WRITEABLE"] = False
            return a
        case "memoryview":
            # Numeric/plain dtypes → create a readonly buffer-backed view that can't be flipped.
            a_c = np.ascontiguousarray(a)
            ro_buf = memoryview(a_c).toreadonly()
            return np.frombuffer(ro_buf, dtype=a_c.dtype).reshape(a_c.shape)
        case _:
            msg = f"Invalid method: {method!r}"
            raise ValueError(msg)
