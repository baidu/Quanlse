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
Pulse generator for Quanlse Scheduler and Experiment.
"""

from typing import Dict, List, Callable

from Quanlse.Superconduct.Lab.LabSpace import LabSpace
from Quanlse.QWaveform import QJob
from Quanlse.QOperation import Error, CircuitLine
from Quanlse.Utils.ControlGate import ControlGate


class Generator:
    """
    Basic class of pulse Generator for Quanlse Scheduler.

    :param labSpace: the LabSpace instance.
    """
    def __init__(self, labSpace: LabSpace):
        """ Initialize """
        self.labSpace = labSpace  # type:LabSpace
        self._generators = {}  # type: Dict[str, Callable]

    def __call__(self, *args, **kwargs) -> QJob:
        """
        Call the instance and generate pulses.
        """
        if isinstance(args[0], CircuitLine):
            # cirLine = args[0]  # type: CircuitLine
            # gateName = cirLine.data.name
            # qubitIndex = list(cirLine.qRegIndexList)
            raise Error.RuntimeError("Not implemented for `CircuitLine` type of gates.")
        elif isinstance(args[0], ControlGate):
            ctrlGate = args[0]  # type: ControlGate
            gateName = ctrlGate.name
            qubitIndex = list([ctrlGate.onQubit] if isinstance(ctrlGate.onQubit, int) else ctrlGate.onQubit)
            qubits = "".join(self.labSpace.getQubitLabel(qubitIndex))
        else:
            raise Error.RuntimeError(f"Can not analysis input gate of type `{type(args[0])}`.")

        # Read from labSpace
        configKey = f"gates.{qubits}.{gateName}"
        presetJob = self.labSpace.getConfig(configKey)
        if presetJob is not None:
            # The gate is preset in LabSpace
            job = QJob.parseJson(presetJob)
            return job
        else:
            if gateName in self.generators.keys():
                return self.generators[gateName](gate=gateName, qubitIndex=qubitIndex, labSpace=self.labSpace)
            else:
                raise Error.RuntimeError(f"Gate `{gateName}` is not supported by the generator.")

    @property
    def generators(self) -> Dict[str, Callable]:
        return self._generators

    def addGenerator(self, gateList: List[str], generator: Callable) -> None:
        """
        Add pulse generator.

        :param gateList: a gate list to add the generator to
        :param generator: generator to be added
        :return: None
        """
        for gateName in gateList:
            self.generators[gateName] = generator
