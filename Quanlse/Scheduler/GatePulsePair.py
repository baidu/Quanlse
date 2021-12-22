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
The GatePulsePair object.
"""

import copy
from typing import List, Union, Optional

from Quanlse.QWaveform import QJob
from Quanlse.QOperator import QOperator
from Quanlse.QPlatform import Error
from Quanlse.QOperation.FixedGate import FixedGateOP
from Quanlse.QOperation.RotationGate import RotationGateOP


class GatePulsePair:
    """
    A gate with its corresponding pulses.

    :param gate: the FixedGateOP or RotationGateOP object.
    :param qubits: indicates the qubits of the gate performing on
    :param job: the QJob object
    :param t0: the start time
    """

    def __init__(self, gate: Union[FixedGateOP, RotationGateOP] = None, qubits: Union[int, List[int]] = None,
                 t0: float = None, job: QJob = None):
        """ Initialization """
        # Check property
        if gate is not None:
            if gate.bits is not None:
                if isinstance(qubits, int):
                    if gate.bits != 1:
                        raise Error.ArgumentError(f"The gate performs on {gate.bits} qubits, however onSysNum "
                                                  f"performs on only 1 qubits.")
                elif isinstance(qubits, list):
                    if gate.bits != len(qubits):
                        raise Error.ArgumentError(f"The gate performs on {gate.bits} qubits, however onSysNum "
                                                  f"performs on {len(qubits)} qubits.")
        # Set properties
        self._gate = gate  # type: Union[FixedGateOP, RotationGateOP]
        self._qubits = qubits  # type: Union[int, List[int]]
        self._onSubSys = []  # type: List[int]
        self._t = 0.0  # type: float
        self._t0 = t0  # type: float
        self.job = job

    @property
    def job(self) -> Union[QJob, None]:
        """
        Get pulses property
        """
        return self._job

    @job.setter
    def job(self, item: Optional[QJob]):
        """
        Set pulses property
        """
        if item is not None:
            self._job = copy.deepcopy(item)
            # pulse duration
            self._t, _ = self.job.computeMaxTime()
            # Generate onSubSys
            onSubSys = []
            for opKey in self.job.ctrlOperators.keys():
                ops = self.job.ctrlOperators[opKey]
                if isinstance(ops, list):
                    for op in ops:
                        onSubSys.append(op.onSubSys)
                elif isinstance(ops, QOperator):
                    onSubSys.append(ops.onSubSys)

            _onSubSys = onSubSys if isinstance(onSubSys, list) else [onSubSys]
            _gateQubits = self._qubits if isinstance(self._qubits, list) else [self._qubits]
            onSubSys = list(set(_onSubSys) | set(_gateQubits))
            onSubSys.sort()
            self._onSubSys = onSubSys
        else:
            self._job = None
            self._onSubSys = self._qubits
            self._t = 0

    @property
    def qubits(self) -> Union[int, List[int]]:
        """
        Indicates the qubits of the gate performing on
        """
        return self._qubits

    @property
    def t(self) -> float:
        """
        Pulse duration
        """
        return self._t

    @property
    def t0(self) -> float:
        """
        The start time of the GatePulsePair
        """
        return self._t0

    @t0.setter
    def t0(self, value: float):
        """
        The start time of the GatePulsePair
        """
        self._t0 = value

    @property
    def onSubSys(self) -> List[int]:
        """
        The sub-systems which the pulses work on.
        """
        return self._onSubSys

    @property
    def gate(self) -> Union[FixedGateOP, RotationGateOP]:
        """
        Get the FixedGateOP or RotationGateOP object
        """
        return self._gate

    @gate.setter
    def gate(self, item: Union[FixedGateOP, RotationGateOP]):
        """
        Set the FixedGateOP or RotationGateOP object
        """
        self._gate = copy.deepcopy(item)

    def hasPulse(self):
        """
        check if the instance is empty
        """
        return True if self.job is not None else False
