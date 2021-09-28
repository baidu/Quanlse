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

import sys
import copy
from typing import List, Union, Optional

from Quanlse.QOperation.FixedGate import FixedGateOP
from Quanlse.QOperation.RotationGate import RotationGateOP
from Quanlse.QWaveform import QJob
from Quanlse.QOperation import Error, CircuitLine
from Quanlse.QOperator import QOperator


class GatePulsePair:
    """
    A gate with its corresponding pulses.

    :param cirLine: quantum gate in the quantum circuit model.
    :param pulses: QJob object with corresponding pulses.
    """

    def __init__(self, cirLine: CircuitLine, pulses: QJob = None):
        """ Initialization """
        # Set properties
        self._pulses = None  # type: Optional[QJob]
        self._cirLine = None  # type: Optional[CircuitLine]
        self._onSubSys = []  # type: List[int]
        self._t = 0.0  # type: float

        # Set objects
        self.cirLine = copy.deepcopy(cirLine)
        self.pulses = copy.deepcopy(pulses)

    def __eq__(self, other):
        """ Operator == """
        return self._compareObj(other)

    def __ne__(self, other):
        """ Operator != """
        return not self._compareObj(other)

    @property
    def pulses(self) -> Union[QJob, None]:
        """
        Get pulses property
        """
        return self._pulses

    @pulses.setter
    def pulses(self, item: Optional[QJob]):
        """
        Set pulses property
        """
        if item is not None:
            self._pulses = copy.deepcopy(item)
            # pulse duration
            self._t, _ = self.pulses.computeMaxTime()
            # Generate onSubSys
            onSubSys = []
            for opKey in self.pulses.ctrlOperators.keys():
                ops = self.pulses.ctrlOperators[opKey]
                if isinstance(ops, list):
                    for op in ops:
                        onSubSys.append(op.onSubSys)
                elif isinstance(ops, QOperator):
                    onSubSys.append(ops.onSubSys)

            onSubSys = list(set(onSubSys) | set(self.cirLine.qRegIndexList))
            onSubSys.sort()
            self._onSubSys = onSubSys

    @property
    def t(self) -> float:
        """
        Pulse duration
        """
        return self._t

    @property
    def onSubSys(self) -> List[int]:
        """
        The sub-systems which the pulses work on.
        """
        return self._onSubSys

    @property
    def cirLine(self):
        """
        Get cirLine property
        """
        return self._cirLine

    @cirLine.setter
    def cirLine(self, item):
        """
        Set cirLine property
        """
        self._cirLine = copy.deepcopy(item)

    def _compareObj(self, other):
        """
        Compare two GatePulsePair object.
        """
        if not isinstance(other, GatePulsePair):
            raise Error.ArgumentError(f"GatePulsePair object can not compare with a {type(other)}.")
        if isinstance(other.cirLine.data, FixedGateOP) and isinstance(self.cirLine.data, RotationGateOP):
            return False
        if isinstance(other.cirLine.data, RotationGateOP) and isinstance(self.cirLine.data, FixedGateOP):
            return False
        if other.cirLine.data.name != self.cirLine.data.name:
            return False
        if other.cirLine.qRegIndexList != self.cirLine.qRegIndexList:
            return False
        if isinstance(self.cirLine.data, RotationGateOP):
            argLen = len(other.cirLine.data.uGateArgumentList)
            for idx in range(argLen):
                verify = abs(other.cirLine.data.uGateArgumentList[idx] -
                             self.cirLine.data.uGateArgumentList[idx])
                if verify > sys.float_info.epsilon:
                    return False
        return True

    def hasPulse(self):
        """
        check if the instance is empty
        """
        return True if self.pulses is not None else False