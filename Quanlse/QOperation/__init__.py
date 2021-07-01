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
Quantum Operation
"""
from typing import List, Union, Optional, Callable, TYPE_CHECKING

import numpy

from Quanlse.QPlatform import Error

if TYPE_CHECKING:
    from Quanlse.QOperation.FixedGate import FixedGateOP
    from Quanlse.QOperation.RotationGate import RotationGateOP
    from Quanlse.QRegPool import QRegStorage

OperationFunc = Callable[[*'QRegStorage'], None]
RotationArgument = float


class QOperation:
    """
    Basic classes for quantum operation
    """

    def __init__(self, name: Optional[str] = None, bits: Optional[int] = None,
                 matrix: Optional[numpy.ndarray] = None) -> None:
        """
        Constructor for QOperation class
        """
        self.name = name
        self.bits = bits
        self._matrix = matrix

    def getMatrix(self) -> numpy.ndarray:
        """
        Returns a numpy ndarray

        :return: returned matrix in ndarray
        """
        if self.__class__.__name__ == 'FixedGateOP':
            return self._matrix
        elif self.__class__.__name__ == 'RotationGateOP':
            if self._matrix is None:
                self._matrix = self.generateMatrix()
            return self._matrix
        elif self.__class__.__name__ == 'CustomizedGateOP':
            return self._matrix
        else:
            raise Error.ArgumentError(f'{self.__class__.__name__} do not have matrix!')

    def _op(self, qRegList: List['QRegStorage']) -> None:
        """
        Quantum operation base

        :param qRegList: quantum register list
        :return: None
        """
        env = qRegList[0].env
        for qReg in qRegList:
            if qReg.env != env:
                raise Error.ArgumentError('QReg must belong to the same env!')

        if env.__class__.__name__ == 'QProcedure':
            raise Error.ArgumentError('QProcedure should not be operated!')

        if self.bits is not None and self.bits != len(
                qRegList):  # Barrier and QProcedure does not match bits configuration
            raise Error.ArgumentError('The number of QReg must match the setting!')

        if len(qRegList) <= 0:
            raise Error.ArgumentError('Must have QReg in operation!')

        if len(qRegList) != len(set(qReg for qReg in qRegList)):
            raise Error.ArgumentError('QReg of operators in circuit are not repeatable!')

        circuitLine = CircuitLine()
        circuitLine.data = self
        circuitLine.qRegIndexList = [qReg.index for qReg in qRegList]
        env.circuit.append(circuitLine)


Operation = Union[
    'FixedGateOP', 'RotationGateOP']


class CircuitLine:
    """
    This class defines a quantum gate in the quantum circuit model.
    It specifies two key components to characterize a quantum gate:
    The gate and its operated qubit indices.
    """
    data: None  # type: Operation
    qRegIndexList: None  # type: List[int]

    def __init__(self, data: Operation = None, qRegIndexList: List[int] = None):
        r"""
        Initialize a quantum gate instance.

        :param data: a Quanlse.QOperation.Operation instance,
                    the quantum gate to be applied.
        :param qRegIndexList: a list of qubit indices.
                    If `gate` is a single-qubit
                    gate, then `qubits` still be a List of the form `[i]`
        """
        self.data = data
        self.qRegIndexList = qRegIndexList
