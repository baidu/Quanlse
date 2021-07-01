#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2021 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities
"""

from typing import Dict, List, Optional, Union, Type

import pickle
import base64
import numpy

from Quanlse.QPlatform import Error
from Quanlse.QOperation import CircuitLine


def numpyMatrixToDictMatrix(numpyMatrix: numpy.ndarray) -> Dict:
    """
    Convert numpy matrix to a dictionary. Must be C-contiguous.

    :param numpyMatrix: numpy matrix
    :return: returned dictionary
    """

    if not numpyMatrix.flags["C_CONTIGUOUS"]:
        raise Error.ArgumentError('Matrix must C-contiguous!')

    if numpyMatrix.size == 0:
        return {}

    array = []
    dictMatrix = {
        'shape': list(numpyMatrix.shape),
        'array': array
    }

    for value in numpy.nditer(numpyMatrix):
        val = value.reshape(1, 1)[0][0]
        array.append({
            'real': val.real,
            'imag': val.imag
        })
    return dictMatrix


def dictMatrixToNumpyMatrix(dictMatrix: Dict, valueType: Union[Type[complex], Type[float]]) -> numpy.ndarray:
    """
    Convert numpy matrix to a dictionary. Must be C-contiguous.

    :param dictMatrix: dictionary consisting the matrix
    :param valueType: select value type
    :return: returned numpy matrix
    """

    if len(dictMatrix) == 0:
        return numpy.empty(0, valueType)

    if valueType == complex:
        complexArray = [complex(complexValue['real'], complexValue['imag']) for complexValue in dictMatrix['array']]
    else:
        complexArray = [complexValue['real'] for complexValue in dictMatrix['array']]
    return numpy.array(complexArray).reshape(dictMatrix['shape'])


def circuitLineDump(cirLine: Union[CircuitLine, List[CircuitLine]]) -> str:
    """
    Convert CircuitLine Object (or the list of that) into a string.

    :param cirLine: a CircuitLine Object
    :return: returned string object
    """
    byteStr = pickle.dumps(cirLine)
    base64str = base64.b64encode(byteStr)
    return base64str.decode()


def circuitLineLoad(base64str: str) -> Union[CircuitLine, List[CircuitLine]]:
    """
    Convert base64 string into CircuitLine objects.
    
    :param base64str: input base64 string
    :return: returned CircuitLine object
    """
    byteStr = base64.b64decode(base64str.encode())
    obj = pickle.loads(byteStr)  # type: Union[CircuitLine, List[CircuitLine]]
    return obj