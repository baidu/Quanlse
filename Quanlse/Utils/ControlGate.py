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
Experiment class for superconducting system
"""

from numpy import ndarray
from typing import List, Tuple, Union, Any, Dict, Optional

from Quanlse.QWaveform import QJob
from Quanlse.Superconduct.Lab.Utils import Scan


class ControlGate(object):
    """
    The quantum gate instance with the gate definition as well as the control pulse.
    """

    def __init__(self, name: str, onQubit: Union[int, List[int]] = None, job: QJob = None):
        """ Initialization """
        self._name = name  # type: str
        self._matrix = None  # type: Optional[ndarray]

        # Set properties
        self._onQubit = onQubit  # type: Optional[Union[int, List[int]]]
        self._job = job  # type: Optional[QJob]

    @property
    def name(self) -> str:
        """ The qubit name(s) the gate performs on """
        return self._name

    @name.setter
    def name(self, name: str):
        """ The qubit name(s) the gate performs on """
        self._name = name

    @property
    def job(self) -> QJob:
        """ Get the QJob instance """
        return self._job

    @job.setter
    def job(self, job: QJob):
        """ Get the QJob instance """
        self._job = job

    @property
    def onQubit(self) -> Optional[Union[int, List[int]]]:
        """ The qubit name(s) the gate performs on """
        return self._onQubit

    @onQubit.setter
    def onQubit(self, onQubit: Optional[Union[int, List[int]]]):
        """ The qubit name(s) the gate performs on """
        self._onQubit = onQubit

    def scanDim(self) -> int:
        """ Return the scanning dimensional of the current gate """
        if self.job is None:
            return 0
        _scanDimCount = 0
        for _waveKey in self.job.waves.keys():
            for _waveId, _wave in enumerate(self.job.waves[_waveKey]):
                _waveDict = _wave.dump2Dict()
                for _settingKey in _waveDict.keys():
                    if isinstance(_waveDict[_settingKey], Scan):
                        _scanDimCount += 1
                for _argKey in _waveDict['args'].keys():
                    if isinstance(_waveDict['args'][_argKey], Scan):
                        _scanDimCount += 1
        return _scanDimCount
