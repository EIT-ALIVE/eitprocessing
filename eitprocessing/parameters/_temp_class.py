from ..binreader import MaxValue
from ..binreader import MinValue
from ..binreader import Sequence


class DetectBreaths:
    def __init__(self, method: str = "Extreme values"):
        self.method = method

    def apply(self, sequence: Sequence) -> list[tuple[int, int, int]]:
        min_phases = list(
            filter(lambda phase: isinstance(phase, MinValue), sequence.phases)
        )
        max_phases = list(
            filter(lambda phase: isinstance(phase, MaxValue), sequence.phases)
        )

        while max_phases[0].index < min_phases[0].index:
            max_phases = max_phases[1:]

        while min_phases[-1].index < max_phases[-1].index:
            max_phases = max_phases[:-1]

        breaths = []
        for start, middle, end in zip(min_phases, max_phases, min_phases[1:]):
            breaths.append((start.index, middle.index, end.index))

        return breaths
