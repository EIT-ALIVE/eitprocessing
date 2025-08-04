import warnings
from collections.abc import Hashable
from dataclasses import dataclass, field, replace
from typing import TypeVar, cast, overload

import numpy as np
from frozendict import frozendict
from typing_extensions import Self

from eitprocessing.datahandling.eitdata import EITData
from eitprocessing.datahandling.pixelmap import PixelMap
from eitprocessing.roi import PixelMask

T = TypeVar("T", bound=EITData | PixelMap)


@dataclass(frozen=True)
class PixelMaskCollection:
    """A collection of pixel masks, each representing a specific region of interest (ROI) in the EIT data.

    This class allows for the application of multiple masks to numpy arrays, EIT data or pixel maps, enabling the
    extraction of specific regions based on the defined masks.

    PixelMasks can be labelled with strings, or anonymous (`label=None`). Either all or none of the masks should have a
    label; it is not possible to mix labelled and unlabelled masks.

    At initialization, masks can be provided as a list or a dictionary. If provided as a list, the masks are converted
    to a dictionary with either their labels as keys (if all masks have labels) or their indices as keys (if none have
    labels). If provided as a dictionary, the keys must match the labels of the masks; anonymous PixelMaps can not be
    provided as a dictionary.

    When the `apply` method is called, it applies each mask to the provided data and returns a dictionary
    mapping each mask's key (label or index) to the resulting masked data.

    Example:
    >>> mask_collection = PixelMaskCollection(
            [
                PixelMask(right_lung_data, label="right lung"),
                PixelMask(left_lung_data, label="left lung"),
            ],
            label="lung masks",
        )

    Args:
        masks (dict): A dictionary mapping mask names to their corresponding pixel masks.
        label (str | None): An optional label for the collection of masks.

    Attributes:
        is_anonymous (bool):
            A boolean indicating whether all masks in the collection are anonymous (i.e., have no label).

    """

    masks: frozendict[Hashable, PixelMask]
    label: str | None = field(default=None)

    def __init__(
        self,
        masks: dict[Hashable, PixelMask] | frozendict[Hashable, PixelMask] | list[PixelMask],
        *,
        label: str | None = None,
    ):
        object.__setattr__(self, "masks", self._validate_and_convert_input_masks(masks))
        object.__setattr__(self, "label", label)

    def _validate_and_convert_input_masks(
        self, masks: dict[Hashable, PixelMask] | frozendict[Hashable, PixelMask] | list[PixelMask]
    ) -> frozendict[Hashable, PixelMask]:
        """Validate the input masks and convert to a frozendict.

        The input masks are valid if:
        - The input is a list with all labelled or all anonymous PixelMask instances.
        - The input is a dictionary where each PixelMask instance's label matches the key.
        """
        # Check for type before checking for length, e.g., and empty string should raise a TypeError, not ValueError
        if not isinstance(masks, (list, dict, frozendict)):
            msg = f"Expected a list or a dictionary, got {type(masks)}."
            raise TypeError(msg)

        if not masks:
            msg = "A PixelMaskCollection should contain at least one mask."
            raise ValueError(msg)

        if isinstance(masks, list):
            return self._validate_and_convert_list_input(masks)

        return self._validate_and_convert_dict_input(masks)

    def _validate_and_convert_list_input(self, masks: list[PixelMask]) -> frozendict[Hashable, PixelMask]:
        if any(not isinstance(mask, PixelMask) for mask in masks):
            msg = "All items in the collection must be instances of PixelMask."
            raise TypeError(msg)
        if all(mask.label is None for mask in masks):
            return frozendict(enumerate(masks))
        if all(isinstance(mask.label, str) for mask in masks):
            return frozendict({cast("str", mask.label): mask for mask in masks})

        msg = "Cannot mix labelled and anonymous masks in a collection."
        raise ValueError(msg)

    def _validate_and_convert_dict_input(
        self, masks: dict[Hashable, PixelMask] | frozendict[Hashable, PixelMask]
    ) -> frozendict[Hashable, PixelMask]:
        if any(not isinstance(mask, PixelMask) for mask in masks.values()):
            msg = "All items in the collection must be instances of PixelMask."
            raise TypeError(msg)
        if all(mask.label is None for mask in masks.values()):
            if set(masks.keys()) != set(range(len(masks))):
                msg = "Anonymous masks should be indexed with consecutive integers starting from 0."
                raise ValueError(msg)
        elif any(mask.label != key for key, mask in masks.items()):
            msg = "Keys should match the masks' label."
            raise KeyError(msg)
        return frozendict(masks)

    @property
    def is_anonymous(self) -> bool:
        """Check if all masks in the collection are anonymous (i.e., have no label)."""
        return all(mask.label is None for mask in self.masks.values())

    def add(self, *mask: PixelMask, overwrite: bool = False, **kwargs) -> Self:
        """Return a new collection with one or more masks added.

        Masks can be added as positional arguments or keyword arguments (non-anonymous masks only).

        Example:
        >>> left_lung_mask = PixelMask(data, label="left_lung")
        >>> mask_collection.add(left_lung_mask)  # equals mask_collection.add(left_lung=left_lung_mask)

        Args:
            *mask (PixelMask): One or more `PixelMask` instances to add to the collection.
            overwrite (bool): If True, allows overwriting existing masks with the same label or index.
            **kwargs (PixelMask):
                Keyword arguments where the key is the label of the mask and the value is the mask itself.

        Returns:
        A new `PixelMaskCollection` instance with the added masks.

        Raises:
            ValueError:
                If no masks are provided, if trying to mix labelled and unlabelled masks, or if provided keys don't
                match the masks label.
            KeyError: If trying to overwrite an existing mask without setting `overwrite`.
        """
        if not mask and not kwargs:
            msg = "No masks provided to add."
            raise ValueError(msg)

        if self.is_anonymous:
            return self._add_to_anonymous(mask, kwargs, overwrite)

        return self._add_to_labelled(mask, kwargs, overwrite)

    def _add_to_anonymous(self, masks: tuple[PixelMask, ...], kwargs: dict[str, PixelMask], overwrite: bool) -> Self:
        if overwrite:
            warnings.warn(
                "Cannot overwrite existing masks in an anonymous collection. All masks with be added instead.",
                UserWarning,
            )
        if kwargs:
            msg = "Cannot mix labelled and anonymous masks in a collection."
            raise ValueError(msg)
        if not all(mask_.label is None for mask_ in masks):
            msg = "Cannot mix labelled and anonymous masks in a collection."
            raise ValueError(msg)
        return self.update(masks=self.masks | dict(enumerate(masks, start=len(self.masks))))

    def _add_to_labelled(self, masks: tuple[PixelMask, ...], kwargs: dict[str, PixelMask], overwrite: bool) -> Self:
        new_masks: dict[str, PixelMask] = {}

        if masks:
            if any(mask.label is None for mask in masks):
                msg = "Cannot mix labelled and anonymous masks in a collection."
                raise ValueError(msg)

            if not overwrite and any(mask_.label in self.masks for mask_ in masks):
                msg = "Cannot overwrite mask with the same key unless 'overwrite' is set to True."
                raise KeyError(msg)

            new_masks = new_masks | {cast("str", mask.label): mask for mask in masks}

        if kwargs:
            if not overwrite and any(key in self.masks for key in kwargs):
                msg = "Cannot overwrite mask with the same key unless 'overwrite' is set to True."
                raise KeyError(msg)
            new_masks = new_masks | kwargs

        return self.update(masks=self.masks | new_masks)

    update = replace  # dataclasses.replace

    @overload
    def apply(self, data: np.ndarray) -> dict[Hashable, np.ndarray]: ...

    @overload
    def apply(self, data: EITData, *, label_format: str | None = None, **kwargs) -> dict[Hashable, EITData]: ...

    @overload
    def apply(self, data: PixelMap, *, label_format: str | None = None, **kwargs) -> dict[Hashable, PixelMap]: ...

    def apply(self, data, *, label_format: str | None = None, **kwargs):
        """Apply the masks to the provided data.

        The input data can be a numpy array, EITData, or PixelMap. The method applies each mask in the collection to the
        data and returns a dictionary mapping each mask's key (label or index) to the resulting masked data.

        If `label_format` is provided, it should be a format string that includes `{mask_label}`. This label will be
        passed to the resulting objects, with the appropriate mask label applied. If `label_format` is not provided, no
        label will be provided.

        Additional keyword arguments are passed to the `update` of the EITData or PixelMap, if applicable. If the input
        data is a numpy array, `label_format` and additional keyword arguments are not applicable and will raise a
        `ValueError`.

        Example:
        >>> mask_collection = PixelMaskCollection([
                PixelMask(data, label="right lung"),
                PixelMask(data, label="left lung")
            ])
        >>> mask_collection.apply(eit_data, label_format="masked {mask_label}")
        {"right lung": EITData(label="masked right lung"), "left lung": EITData(label="masked left lung")}

        Args:
            data (np.ndarray | EITData | PixelMap): The data to which the masks will be applied.
            label_format (str | None): A format string to create the label of the returned EITData or PixelMap objects.
            **kwargs:
                Additional keyword arguments to pass to the `update` method of the returned EITData or PixelMap objects.

        Returns:
            A dictionary mapping each mask's key (label or index) to the resulting masked data.

        Raises:
            ValueError: If a label is passed as a keyword argument.
            ValueError:
                If label_format or additional keyword arguments are provided when the input data is a numpy array.
            ValueError: If provided label format does not contain '{mask_label}'.
            TypeError: If provided data is not an array, EITData, or PixelMap.
        """
        if "label" in kwargs:
            msg = "Cannot pass 'label' as a keyword argument to `apply()`. Use 'label_format' instead."
            raise ValueError(msg)

        match data:
            case np.ndarray():
                return self._apply_mask_array(data, label_format, **kwargs)
            case EITData():
                return self._apply_mask_data(data, label_format, **kwargs)
            case PixelMap():
                # The extra case seems superfluous, but is required for type checking; _apply_mask_data() returns a dict
                # with the values the same type as the input data, which is not clear if the cases EITData and PixelMap
                # are not separated.
                return self._apply_mask_data(data, label_format, **kwargs)
            case _:
                msg = f"Unsupported data type: {type(data)}"
                raise TypeError(msg)

    def _apply_mask_array(self, data: np.ndarray, label_format: str | None, **kwargs) -> dict[Hashable, np.ndarray]:
        if label_format is not None:
            msg = "label_format is not applicable for numpy arrays."
            raise ValueError(msg)
        if kwargs:
            msg = "Additional keyword arguments are not applicable for numpy arrays."
            raise ValueError(msg)
        return {key: mask.apply(data) for key, mask in self.masks.items()}

    def _apply_mask_data(self, data: T, label_format: str | None, **kwargs) -> dict[Hashable, T]:
        def get_label(key: Hashable) -> str | None:
            """Format the label with the mask_label, provided `label_format` is not None."""
            if label_format is None:
                return None
            try:
                formatted_label = label_format.format(mask_label=key)
            except IndexError as e:
                msg = f"Invalid label format: {label_format}."
                raise ValueError(msg) from e
            except KeyError as e:
                msg = f"Invalid label format: {label_format}."
                raise ValueError(msg) from e
            if formatted_label == label_format:
                msg = "Invalid label format. Label format does not contain '{mask_label}'. Cannot format label."
                raise ValueError(msg)
            return formatted_label

        return {key: mask.apply(data, label=get_label(key), **kwargs) for key, mask in self.masks.items()}
