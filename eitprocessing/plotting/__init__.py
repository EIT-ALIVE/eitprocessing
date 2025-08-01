from typing import Protocol, cast, runtime_checkable

from eitprocessing.datahandling.pixelmap import (
    DifferenceMap,
    ODCLMap,
    PendelluftMap,
    PerfusionMap,
    PixelMap,
    SignedPendelluftMap,
    TIVMap,
)
from eitprocessing.plotting.pixelmap import (
    Config,
    DifferenceMapPlotConfig,
    ODCLMapPlotConfig,
    PendelluftMapPlotConfig,
    PerfusionMapPlotConfig,
    PixelMapPlotConfig,
    SignedPendelluftMapPlotConfig,
    TIVMapPlotConfig,
)


@runtime_checkable
class HasPlottableConfig(Protocol):
    """Protocol for objects that have an immutable plot configuration."""

    @property  # Stated as a property in the protocol to make it immutable.
    def _plot_config(self) -> Config: ...


_PLOT_CONFIG_REGISTRY: dict[type[HasPlottableConfig], Config] = {
    PixelMap: PixelMapPlotConfig(),
    TIVMap: TIVMapPlotConfig(),
    ODCLMap: ODCLMapPlotConfig(),
    DifferenceMap: DifferenceMapPlotConfig(),
    PerfusionMap: PerfusionMapPlotConfig(),
    PendelluftMap: PendelluftMapPlotConfig(),
    SignedPendelluftMap: SignedPendelluftMapPlotConfig(),
}


def get_plot_config(obj: HasPlottableConfig | type[HasPlottableConfig]) -> Config:
    """Get the appropriate plot configuration for a given type.

    Args:
        obj (object or type):
            The instance or type for which to get the plot configuration.

    Returns:
        Config: The plot configuration specific to the type.
    """
    type_ = type(obj) if not isinstance(obj, type) else obj
    type_ = cast("type[HasPlottableConfig]", type_)

    try:
        return _PLOT_CONFIG_REGISTRY[type_]
    except KeyError as e:
        msg = f"No plot configuration registered for type {type_}."
        raise ValueError(msg) from e


def set_plot_config(type_: type[HasPlottableConfig], config: Config) -> None:
    """Register a plot configuration for a specific type.

    This overwrites earlier configurations for the type.

    Args:
        type_ (HasPlottableConfig): The type for which to set the configuration.
        config (Config): The configuration instance to register.
    """
    _PLOT_CONFIG_REGISTRY[type_] = config


def set_plot_config_parameters(*types: type[HasPlottableConfig], **parameters) -> None:
    """Set or update the plot configuration for specified types.

    Examples:
        >>> set_plot_config(TIVMap, cmap="plasma")
        >>> set_plot_config(PendelluftMap, SignedPendelluftMap, colorbar=False, absolute=True)
        >>> set_plot_config(cmap="viridis")  # Update all types with new cmap; requires all types to have this parameter

    """
    if not types:
        types = tuple(_PLOT_CONFIG_REGISTRY.keys())

    for type_ in types:
        _PLOT_CONFIG_REGISTRY[type_] = _PLOT_CONFIG_REGISTRY[type_].update(**parameters)


def reset_plot_config(*types: type[HasPlottableConfig]) -> None:
    """Reset plot config to their defaults.

    Resets the plot config defaults for the specified types. If no types are specified, all registered types will be
    reset to their default config.
    """
    if not types:
        types = tuple(_PLOT_CONFIG_REGISTRY.keys())
    for type_ in types:
        _PLOT_CONFIG_REGISTRY[type_] = _PLOT_CONFIG_REGISTRY[type_].__class__()
