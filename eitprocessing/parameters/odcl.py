from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from eitprocessing.datahandling.pixelmap import ODCLMap
from eitprocessing.parameters.tidal_impedance_variation import TIV
from eitprocessing.roi import PixelMask

if TYPE_CHECKING:
    from eitprocessing.datahandling.eitdata import EITData


@dataclass(frozen=True, kw_only=True)
class OdclPeepTrial:
    """Calculate relative overdistention and collapse from a series of PEEP steps.

    Provided a list of EITData objects from a decremental PEEP trial, calculates the relative overdistention and
    collapse according to Costa et al. (2009).
    """

    compliance_method: Literal["constant driving pressure"] = "constant driving pressure"

    def apply(
        self,
        eit_datasets: list[EITData],
    ) -> OdclPeepTrialResult:
        """Apply the algorithm to the provided datasets.

        The provided datasets should be in decremental PEEP order, i.e. in order of measurement.

        Returns an object containing an ODCLMap, overdistention an collapse value per provided PEEP step.

        Arguments:
            eit_datasets (list or EITData): EITData objects

        Returns:
            OdclPeepTrialResult: object containing results and plotting functionality
                odcl_maps: list of ODCLMap
                overdistention: array of overdistention values
                collapse: array of collapse values
                plotting: plotting functionality

        """
        n_steps = len(eit_datasets)
        continuous_datasets = [eit_data.get_summed_impedance() for eit_data in eit_datasets]

        mask = PixelMask(
            ~np.all(
                [np.all(np.isnan(eit_data.pixel_impedance), axis=0) for eit_data in eit_datasets],
                axis=0,
            )
        )  # infer mask from all-NaN pixels

        tiv = TIV()
        tiv_per_breath = [
            tiv.compute_pixel_parameter(
                eit_data,
                continuous_data=continuous_data,
                sequence=None,
                store=False,
            )
            for eit_data, continuous_data in zip(eit_datasets, continuous_datasets, strict=True)
        ]

        mean_tiv_per_step = np.asarray([np.nanmean(maps.values, axis=0) for maps in tiv_per_breath])
        masked_tiv_per_step = mask.apply(mean_tiv_per_step)

        if self.compliance_method == "constant driving pressure":
            compliance = masked_tiv_per_step
        else:
            msg = f"Unknown compliance method '{self.compliance_method}'"
            raise ValueError(msg)

        compliance = np.clip(compliance, 0, np.inf)

        per_pixel_max_compliance_idx = np.argmax(compliance, axis=0)[np.newaxis]

        max_compliance_per_pixel = np.take_along_axis(
            compliance,
            per_pixel_max_compliance_idx,
            axis=0,
        )

        normalized_max_compliance = max_compliance_per_pixel / np.nanmax(max_compliance_per_pixel)
        relative_compliance = compliance / max_compliance_per_pixel
        relative_compliance_change = (1 - relative_compliance) * normalized_max_compliance

        steps = np.arange(n_steps)[:, np.newaxis, np.newaxis]

        overdistention = np.where(
            steps < per_pixel_max_compliance_idx,
            relative_compliance_change,
            0,
        )
        collapse = np.where(
            steps > per_pixel_max_compliance_idx,
            relative_compliance_change,
            0,
        )

        total_overdistention = np.nansum(overdistention, axis=(1, 2)) / np.nansum(normalized_max_compliance)
        total_collapse = np.nansum(collapse, axis=(1, 2)) / np.nansum(normalized_max_compliance)

        odcl_maps = [ODCLMap(map_) for map_ in mask.apply(overdistention - collapse)]

        return OdclPeepTrialResult(
            odcl_maps=odcl_maps,
            overdistention=total_overdistention,
            collapse=total_collapse,
        )


@dataclass(kw_only=True, frozen=True)
class OdclPeepTrialResult:
    """Result object for the OdclPeepTrial algorithm.

    Contains an ODCLMap, overdistention value and collapse value per PEEP step.

    Attributes:
        odcl_maps: list of ODCLMap pixel maps
        overdistention: array of floats
        collapse: array of floats
        plotting: plotting functionality
            plot_overdistention_collapse:
                plot the overdistention and collapse values in a line plot
            plot_odcl_maps:
                show the ODCL maps in a grid
    """

    odcl_maps: list[ODCLMap]
    overdistention: np.ndarray
    collapse: np.ndarray

    @property
    def plotting(self):
        """Plotting functionality for ODCL PEEP Trial results."""
        from eitprocessing.plotting.odcl import OdclPeepTrialPlotting

        return OdclPeepTrialPlotting(self)
