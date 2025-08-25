# %%
from dataclasses import KW_ONLY, dataclass, field, replace


@dataclass(frozen=True)
class PlotParameters:
    cmap: str = "Reds"
    normalize: bool = True


@dataclass(frozen=True)
class A:
    def __post_init__(self):
        if isinstance(self.plot_parameters, dict):
            object.__setattr__(
                self,
                "plot_parameters",
                replace(type(self).__dataclass_fields__["plot_parameters"].default_factory(), **self.plot_parameters),
            )

    _ = KW_ONLY
    plot_parameters: PlotParameters = field(default_factory=PlotParameters)


@dataclass(frozen=True)
class B(A):
    @dataclass(frozen=True)
    class PlotParams(PlotParameters):
        normalize: bool = False

    _ = KW_ONLY
    plot_parameters: PlotParams = field(default_factory=PlotParams)
