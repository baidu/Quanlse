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
from Quanlse import HardwareImplementation, GateTimeDict

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
        self.name = name
        self.bits = bits
        self._matrix = matrix

    def getMatrix(self) -> numpy.ndarray:
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

    def _op(self, qRegList: List['QRegStorage'], gateTime: Optional[float]) -> None:
        """
        Quantum operation base

        :param qRegList: quantum register list
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

        if self.name == 'CR' and env.hardwareImplementation == HardwareImplementation.CZ:
            raise Error.ArgumentError("You've chosen cz. This gate is not consistent with your choice.")
        elif self.name == 'CZ' and env.hardwareImplementation == HardwareImplementation.CR:
            raise Error.ArgumentError("You've chosen cr. This gate is not consistent with your choice.")
        elif self.name == 'CNOT':
            if env.hardwareImplementation == HardwareImplementation.CR:
                from Quanlse.QOperation import FixedGate, RotationGate
                RotationGate.RZ(0.25)(qRegList[0], gateTime=GateTimeDict['RZ'])  # Use original gateTime
                FixedGate.CR(qRegList[0], qRegList[1], gateTime=gateTime)
                RotationGate.RZ(0.25)(qRegList[1], gateTime=GateTimeDict['RZ'])  # Use original gateTime
            elif env.hardwareImplementation == HardwareImplementation.CZ:
                from Quanlse.QOperation import FixedGate
                FixedGate.H(qRegList[1], gateTime=GateTimeDict['H'])
                FixedGate.CZ(qRegList[0], qRegList[1])
                FixedGate.H(qRegList[1], gateTime=GateTimeDict['H'])
            else:
                raise Error.ArgumentError(f'UnImplemented Quanlse hardware {env.hardwareImplementation}')
        elif self.name == 'SWAP':
            from Quanlse.QOperation import FixedGate
            FixedGate.CNOT(qRegList[0], qRegList[1], gateTime=env.gateTimeDict['SWAP'])
            FixedGate.CNOT(qRegList[1], qRegList[0], gateTime=env.gateTimeDict['SWAP'])
            FixedGate.CNOT(qRegList[0], qRegList[1], gateTime=env.gateTimeDict['SWAP'])
        else:
            circuitLine = CircuitLine()
            circuitLine.data = self
            circuitLine.qRegIndexList = [qReg.index for qReg in qRegList]
            if gateTime is not None:
                circuitLine.gateTime = gateTime
            else:
                circuitLine.gateTime = env.gateTimeDict[self.name]
            env.circuit.append(circuitLine)


Operation = Union[
    'FixedGateOP', 'RotationGateOP']


class CircuitLine:
    """
    Circuit Line
    """
    data: None  # type: Operation
    qRegIndexList: None  # type: List[int]
    gateTime: 0  # type: float
