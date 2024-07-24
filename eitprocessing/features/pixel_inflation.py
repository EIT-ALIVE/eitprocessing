"""Dataclass for pixel inflation detection."""

from dataclasses import dataclass, field

import numpy as np

from eitprocessing.datahandling.breath import Breath
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
        sequence: Sequence,
        eitdata_label: str,
        continuousdata_label: str,
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
            eitdata_label: label of eit data to apply the algorithm to
            continuousdata_label: label of the continuous data to use for global breath detection

        Returns:
            np.ndarray, where each element contains a list of PixelInflation objects
        """
        eitdata = sequence.eit_data[eitdata_label]

        bd_kwargs = self.breath_detection_kwargs.copy()
        bd_kwargs["sample_frequency"] = eitdata.framerate
        breath_detection = BreathDetection(**bd_kwargs)
        breaths = breath_detection.find_breaths(sequence.continuous_data[continuousdata_label])

        breath_middle_indices = np.array(
            [
                np.argmax(eitdata.time == middle_time)
                for middle_time in [breath.middle_time for breath in breaths.values]
            ],
        )

        _, rows, cols = eitdata.pixel_impedance.shape

        pixel_inflations = np.empty((rows, cols), dtype=object)
        for row in range(rows):
            for col in range(cols):
                end = []
                middle = []

                if not np.isnan(
                    eitdata.pixel_impedance[:, row, col],
                ).any() and not np.all(
                    eitdata.pixel_impedance[:, row, col] == 0.0,
                ):
                    start = [
                        np.argmin(
                            eitdata.pixel_impedance[
                                breath_middle_indices[i] : breath_middle_indices[i + 1],
                                row,
                                col,
                            ],
                        )
                        + breath_middle_indices[i]
                        for i in range(len(breath_middle_indices) - 1)
                    ]

                    end = [start[i + 1] for i in range(len(start) - 1)]
                    middle = [
                        np.argmax(
                            eitdata.pixel_impedance[start[i] : start[i + 1], row, col],
                        )
                        + start[i]
                        for i in range(len(start) - 1)
                    ]

                    time = eitdata.time

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

        sequence.interval_data.add(
            IntervalData(
                label="pixel_inflation",
                name="Pixel in- and deflation timing as determined by PixelInflation",
                unit=None,
                category="breath",
                intervals=[
                    (time[breath_middle_indices[i]], time[breath_middle_indices[i + 1]])
                    for i in range(len(breath_middle_indices) - 1)
                ],
                values=pixel_inflations,
                parameters={},
                derived_from=[eitdata],
            ),
        )

        return sequence.interval_data["pixel_inflation"]
