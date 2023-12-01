"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to when electrical impedance tomographs are read.
"""

import io
import struct
import warnings
from dataclasses import dataclass
from typing import Any
from typing import TypeVar
import numpy as np
from numpy.typing import NDArray


T = TypeVar("T")
N = TypeVar("N", bound=np.number)


@dataclass
class Reader:
    file_handle: io.BufferedReader

    def read_single(self, type_code: str, cast: type[T], endian: str = None) -> T:
        data = self._read_full_type_code(type_code, endian)
        return cast(data[0])

    def read_list(self, type_code: str, cast: type[T], length: int, endian: str = None) -> list[T]:
        full_type_code = f"{length}{type_code}"
        data = self._read_full_type_code(full_type_code, endian)
        data = [cast(d) for d in data]
        return data

    def read_array(self, type_code: str, cast: type[N], length: int, endian: str = None) -> NDArray[N]:
        full_type_code = f"{length}{type_code}"
        data = self._read_full_type_code(full_type_code, endian)
        return np.array(data, dtype=cast)

    def read_string(self, length=1):
        full_type_code = f"{length}s"
        data = self._read_full_type_code(full_type_code)
        data = data[0].decode().rstrip()
        return data

    def _read_full_type_code(self, full_type_code, endian: str = None) -> tuple[Any, ...]:
        if endian:
            if endian in ['little', 'big']:
                full_type_code = '<' + full_type_code if endian == 'little' else '>' + full_type_code
            else:
                warnings.warn('Endian type not recognized. Allowed values are '
                              '\'little\' and \' big\'')
        data_size = struct.calcsize(full_type_code)
        packed_data = self.file_handle.read(data_size)
        data = struct.unpack(full_type_code, packed_data)
        return data

    def float32(self, endian: str = None) -> float:
        return self.read_single(type_code="f", cast=float, endian=endian)

    def float64(self, endian: str = None) -> float:
        return self.read_single(type_code="d", cast=float, endian=endian)

    def npfloat32(self, length=1, endian: str = None) -> NDArray[np.float32]:
        return self.read_array(type_code="f", cast=np.float32, length=length, endian=endian)

    def npfloat64(self, length=1, endian: str = None) -> NDArray[np.float64]:
        return self.read_array(type_code="d", cast=np.float64, length=length, endian=endian)

    def int32(self, endian: str = None) -> int:
        return self.read_single(type_code="i", cast=int, endian=endian)

    def npint32(self, length=1, endian: str = None) -> NDArray[np.int32]:
        return self.read_array(type_code="i", cast=np.int32, length=length, endian=endian)

    def string(self, length=1) -> str:
        return self.read_string(length=length)

    def unsigned_char(self, endian: str = None) -> int:
        return self.read_single(type_code="B", cast=int, endian=endian)

    def unsigned_short(self, endian: str = None) -> int:
        return self.read_single(type_code="H", cast=int, endian=endian)

    def unsigned_long_long(self, endian: str = None) -> int:
        return self.read_single(type_code="Q", cast=int, endian=endian)
