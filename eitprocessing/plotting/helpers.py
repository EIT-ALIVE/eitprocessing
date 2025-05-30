from matplotlib.ticker import PercentFormatter, ScalarFormatter


class AbsolutePercentFormatter(PercentFormatter):
    """Format numbers as absolute percentages."""

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Format the tick as an absolute percentage."""
        return super().__call__(abs(x), pos)


class AbsoluteScalarFormatter(ScalarFormatter):
    """Format numbers as absolute values."""

    def __call__(self, x: float, pos: int | None = None) -> str:
        """Format the tick as an absolute value."""
        return super().__call__(abs(x), pos)
