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
Pulse generator for Quanlse Scheduler.
"""

from typing import Dict, Any, List, Callable

from Quanlse.QHamiltonian import QHamiltonian as QHam
from Quanlse.QWaveform import QJob
from Quanlse.QOperation import Error, CircuitLine


class SchedulerPulseGenerator:
    """
    Basic class of pulse Generator for Quanlse Scheduler.

    :param ham: a user-given title to the pi object
    """
    def __init__(self, ham: QHam = None):
        self._ham = ham  # type: QHam
        self._generators = {}  # type: Dict[str, Callable]
        self._conf = {}  # type: Dict[str, Any]

    def __call__(self, *args, **kwargs) -> QJob:
        """
        Call the instance and generate pulses.
        """
        if not isinstance(args[0], CircuitLine):
            raise Error.ArgumentError("You should input a circuitLine instance.")
        _cirLine = args[0]  # type: CircuitLine
        _scheduler = args[1]  # type: 'Scheduler'
        # Call the generator according to the input CircuitLine instance, and
        #     input the system Hamiltonian and the configurations.
        if self._ham is None:
            return self[_cirLine.data.name](cirLine=_cirLine, scheduler=_scheduler)
        else:
            return self[_cirLine.data.name](ham=self._ham, cirLine=_cirLine, scheduler=_scheduler)

    def __getitem__(self, gateName: str) -> Callable:
        """
        Get a generator function according to the gateName.
        """
        if gateName in self._generators:
            return self._generators[gateName]
        else:
            raise Error.ArgumentError(f"Gate {gateName} is not supported.")

    @property
    def generators(self) -> Dict[str, Callable]:
        return self._generators

    @property
    def conf(self):
        return self._conf

    def addGenerator(self, gateList: List[str], generator: Callable) -> None:
        """
        Add pulse generator.

        :param gateList: a gate list to add the generator to
        :param generator: generator to be added
        :return: None
        """
        for gateName in gateList:
            self.generators[gateName] = generator
