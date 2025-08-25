# %%
from functools import singledispatch
from types import NoneType


@singledispatch
def test_func(arg) -> None:
    print("no type", arg)


@test_func.register(str)
def _(string: str) -> None:
    print("str type", string)


@test_func.register(int)
def _(integer: int, second_arg: str) -> None:
    print("integer type", integer, second_arg)


@test_func.register(float)
def _(floating: float) -> None:
    print("floating", floating)


@test_func.register(NoneType)
def _(nothing: None = None) -> None:
    print("none")


# %%

test_func(1.1)
test_func(1, "dinges")
test_func("1.1")
test_func([])
test_func(None)

# %%
