"""
Copyright 2023 Netherlands eScience Center and Erasmus University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods related to when electrical impedance tomographs are read.
"""

import os
import struct
from functools import partialmethod
import numpy as np


class Reader:

    file_handle = None

    def __init__(self, file_handle):
        self.file_handle = file_handle

    def read(self, type_code, length=1, cast=None):
        if len(type_code) > 1:
            full_type_code = type_code * length
        else:
            full_type_code = f"{length}{type_code}"
        data_size = struct.calcsize(full_type_code)
        packed_data = self.file_handle.read(data_size)
        data = struct.unpack(full_type_code, packed_data)

        if length == 1:
            data = data[0]
        
        if cast: 
            return cast(data)
        
        return data

    @staticmethod
    def cast_string(data):
        return data[0].decode().rstrip()

    float32 = partialmethod(read, type_code='f', cast=np.float32)
    float64 = partialmethod(read, type_code='d', cast=np.float64)
    int32 = partialmethod(read, type_code='i', cast=np.int32)
    string = partialmethod(read, type_code='s', cast=cast_string)
