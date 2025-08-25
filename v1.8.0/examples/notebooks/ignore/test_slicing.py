# %%
import operator

import numpy as np
from typing_extensions import Self


# %%
class T:
    def __init__(self, a):
        self.a = a

    def __getitem__(self, slice_: slice) -> Self:
        if isinstance(slice_, tuple):
            slice_, interval = slice_
        else:
            interval = "left-closed"

        match interval:
            case "right" | "(]" | "left-open" | "right-closed" | "]]":
                left_closed, right_closed = False, True
            case "left" | "[)" | "right-open" | "left-closed" | "[[":
                left_closed, right_closed = True, False
            case "both" | "[]" | "closed":
                left_closed, right_closed = True, True
            case "neither" | "()" | "open" | "][":
                left_closed, right_closed = False, False
            case _:
                msg = f"Interval type '{interval}' not recognized."
                raise ValueError(msg)

        print(slice_, type(slice_))
        start_fun = operator.ge if left_closed else operator.gt
        end_fun = operator.le if right_closed else operator.lt

        start_index = np.argmax(start_fun(self.a, slice_.start)) if slice_.start else 0
        end_index = len(self.a) - np.argmax(end_fun(self.a[::-1], slice_.stop)) if slice_.stop else len(self.a) + 2
        return T(self.a[start_index:end_index])


# %%

a = T(np.arange(-10, 10.1, 0.5))
v = a[-10:10, "right-closed"].a
print(v)

# %%
