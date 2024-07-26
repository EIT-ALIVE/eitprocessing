"""Dataclass for pixel inflation detection."""

from dataclasses import dataclass, field

import numpy as np

from eitprocessing.datahandling.breath import Breath
from eitprocessing.datahandling.continuousdata import ContinuousData
from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.intervaldata import IntervalData
from eitprocessing.datahandling.sequence import Sequence
from eitprocessing.features.breath_detection import BreathDetection


@dataclass
class PixelInflation:
    """Algorithm for detecting timing of pixel inflation and deflation in pixel impedance data.

    This algorithm detects the position of start inflation, end inflation/start deflation and
    end deflation in pixel impedance data. It uses BreathDetection to find the global start and end
    of inspiration and expiration. These points are then used to find the start/end of pixel
    inflation/deflation in pixel impedance data.

    Examples:
    pi = PixelInflation(sample_frequency=FRAMERATE)
    pixel_inflations = pi.find_pixel_inflations(sequence, eitdata_label='low pass filtered',
    continuousdata_label='global_impedance_(raw)')
    """

    sample_frequency: float
    breath_detection_kwargs: dict = field(default_factory=dict)

    def find_pixel_inflations(
        self,
        eit_data: EITData,
        continuous_data: ContinuousData,
        result_label: str = "pixel inflations",
        sequence: Sequence | None = None,
        store: bool | None = None,
    ) -> IntervalData:
        """Find pixel inflations in the data.

        This methods finds the pixel start/end of inflation/deflation
        based on the global start/end of inspiration/expiration.
        Pixel start of inflation is defined as the local minimum between
        two global end-inspiration points. Pixel end of deflation is defined
        as the local minimum between the consecutive two global end-inspiration
        points. Pixel end of inflation is defined as the local maximum between
        pixel start of inflation and end of deflation.

        Pixel inflations are constructed as a valley-peak-valley combination,
        representing the start of inflation, the end of inflation/start of
        deflation, and end of deflation.

        Args:
            sequence: the sequence that contains the data
            eit_data: eit data to apply the algorithm to
            continuous_data: continuous data to use for global breath detection
            result_label: label of the returned IntervalData object, defaults to `'pixel inflations'`.
            sequence: optional, Sequence that contains the object to detect pixel inflations in,
            and/or to store the result in.
            store: whether to store the result in the sequence, defaults to `True` if a Sequence if provided.

        Returns:
            An IntervalData object containing Breath objects.
        """
        if store is None and sequence:
            store = True

        if store and sequence is None:
            msg = "Can't store the result if not Sequence is provided."
            raise RuntimeError(msg)

        bd_kwargs = self.breath_detection_kwargs.copy()
        bd_kwargs["sample_frequency"] = eit_data.framerate
        breath_detection = BreathDetection(**bd_kwargs)
        breaths = breath_detection.find_breaths(continuous_data)

        middle_times = np.array(
            [
                np.argmax(eit_data.time == middle_time)
                for middle_time in [breath.middle_time for breath in breaths.values]
            ],
        )

        _, rows, cols = eit_data.pixel_impedance.shape

        pixel_inflations = np.empty((rows, cols), dtype=object)
        for row in range(rows):
            for col in range(cols):
                end = []
                middle = []

                if not np.isnan(
                    eit_data.pixel_impedance[:, row, col],
                ).any() and not np.all(
                    eit_data.pixel_impedance[:, row, col] == 0.0,
                ):
                    start = [
                        np.argmin(
                            eit_data.pixel_impedance[
                                middle_times[i] : middle_times[i + 1],
                                row,
                                col,
                            ],
                        )
                        + middle_times[i]
                        for i in range(len(middle_times) - 1)
                    ]

                    end = [start[i + 1] for i in range(len(start) - 1)]
                    middle = [
                        np.argmax(
                            eit_data.pixel_impedance[start[i] : start[i + 1], row, col],
                        )
                        + start[i]
                        for i in range(len(start) - 1)
                    ]

                    time = eit_data.time

                    inflations = [
                        Breath(time[s], time[m], time[e])
                        for s, m, e in zip(
                            start[:-1],
                            middle,
                            end,
                            strict=True,
                        )
                    ]
                else:
                    inflations = []
                pixel_inflations[row, col] = inflations

        pixel_inflations_container = IntervalData(
            label=result_label,
            name="Pixel in- and deflation timing as determined by PixelInflation",
            unit=None,
            category="breath",
            intervals=[(time[middle_times[i]], time[middle_times[i + 1]]) for i in range(len(middle_times) - 1)],
            values=pixel_inflations,
            parameters={},
            derived_from=[eit_data],
        )

        if store:
            sequence.interval_data.add(pixel_inflations_container)

        return pixel_inflations_container
