from abc import ABC
from dataclasses import fields, is_dataclass

import numpy as np
from typing_extensions import Self

from eitprocessing.mixins.equality import Equivalence


class Addition(Equivalence, ABC):
    def __add__(
        self,
        other: Self,
    ) -> Self:
        return self.concatenate(other=other, check_time_consistency=True)

    def concatenate(
        self,
        other: Self,
        check_time_consistency: bool = True,
        **kwargs,
    ) -> Self:
        _ = self.isequivalent(other, raise_=True)

        if is_dataclass(self):
            obj = self._concatenate_dataclasses(other=other, check_time_consistency=check_time_consistency, **kwargs)

        elif isinstance(self, dict):
            obj = self.__class__(self.data_type)
            for key in self:
                obj[key] = self[key].concatenate(other=other[key], check_time_consistency=check_time_consistency)
        else:
            msg = f"No addition has been implemented for type {type(self)}"
            raise NotImplementedError(msg)

        if len(kwargs):
            msg = f"Unused attribute(s): {kwargs}"
            raise AttributeError(msg)

        return obj

    def _concatenate_dataclasses(self, other, check_time_consistency: bool = True, **kwargs) -> Self:
        obj = {}
        for field in fields(self):
            self_field = getattr(self, field.name)
            other_field = getattr(other, field.name)

            if field.name == "label":
                # Overwrite label with argument value if applicable
                obj["label"] = kwargs.pop("label", self_field)

            elif field.name == "loaded":
                # Concatenated data is not loaded by definition
                obj["loaded"] = False

            elif field.name == "derived_from":
                obj["derived_from"] = self_field + other_field + [self, other]

            elif field.name == "time":
                if check_time_consistency and (om := np.min(other.time)) <= (sm := np.max(self.time)):
                    msg = f"{other} (b) starts before {self} (a) ends: {om} ≤ {sm}"
                    raise ValueError(msg)
                obj["time"] = np.concatenate((self.time, other.time))

            elif field.name == "vendor":
                # Vendor information is already caught in class and should not be passed to class
                pass

            elif field.name == "path":
                from eitprocessing.eit_data import EITData

                obj[field.name] = EITData._ensure_path_list(self_field) + EITData._ensure_path_list(other_field)

            elif isinstance(self_field, Addition):
                obj[field.name] = self_field.concatenate(
                    other=other_field,
                    check_time_consistency=check_time_consistency,
                )

            elif isinstance(self_field, dict):
                obj[field.name] = self_field | other_field

            elif isinstance(self_field, np.ndarray):
                obj[field.name] = np.concatenate((self_field, other_field), axis=0)

                # Lock new object if either self or other was locked
                obj[field.name].flags["WRITEABLE"] = self_field.flags["WRITEABLE"] and other_field.flags["WRITEABLE"]

            elif isinstance(self_field, str):
                if self_field == other_field:
                    obj[field.name] = self_field
                else:
                    obj[field.name] = f"Concatenation of {self_field} and {other_field}"

            elif isinstance(self_field, int | float):
                if field.metadata.get("concatenate", None) == "first":
                    obj[field.name] = self_field

                else:
                    obj[field.name] = self_field + other_field

            elif isinstance(self_field, list):
                obj[field.name] = self_field + other_field

        if len(kwargs):
            msg = f"Unused attribute(s): {kwargs}"
            raise AttributeError(msg)

        return self.__class__(**obj)
