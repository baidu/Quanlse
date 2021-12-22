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
Pipeline class for Quanlse Scheduler.
"""

import copy
from typing import List, Union, Dict, Any, Callable, Optional

from Quanlse.QWaveform import QJob
from Quanlse.QPlatform.Error import ArgumentError
from Quanlse.Scheduler.GatePulsePair import GatePulsePair


class SchedulerProcess:
    """
    Scheduler process for SchedulerPipeline instances to perform scheduling strategy.

    :param title: a user-given title to the pi object
    :param processFunc: Callable function to perform scheduling strategies
    """

    def __init__(self, title: str, processFunc: Callable):
        """
        Constructor for the SchedulerProcess class.
        """
        self.title = title
        self._processFunc = processFunc
        self._conf = {}  # type: Dict[str, Any]

    def __call__(self, scheduler: 'Scheduler') -> QJob:
        """
        Call the scheduling function.
        """
        if self._processFunc is None:
            raise ArgumentError("Scheduling function is not set.")
        return self._processFunc(scheduler=scheduler)


class SchedulerPipeline:
    """
    SchedulerPipeline instance is a sequence of SchedulerProcess to perform scheduling strategies.

    :param title: a user-given title to the pi object
    """

    def __init__(self, title: str = ""):
        """
        Initializing
        """
        self._title = title
        self._pipeline = []  # type: List[SchedulerProcess]
        self._scheduler = None  # type: Optional['Scheduler']

    def __call__(self):
        """
        Call the instance and run the pipeline in order,
        and output a QJob instance.
        """
        for idx, process in enumerate(self._pipeline):
            if isinstance(process, Callable):
                process(self.scheduler)
            else:
                raise ArgumentError("Scheduler process is not callable.")

    @property
    def scheduler(self):
        """
        Return the Scheduler reference.
        """
        return self._scheduler

    @scheduler.setter
    def scheduler(self, scheduler: 'Scheduler'):
        """
        Set the scheduler instance reference.
        """
        self._scheduler = scheduler

    @property
    def pipeline(self) -> List[SchedulerProcess]:
        """
        Return jobs.
        """
        return self._pipeline

    def addPipelineJob(self, process: Union[SchedulerProcess, List[SchedulerProcess]]) -> None:
        """
        Add pipeline job in to the job list
        """
        if isinstance(process, List):
            for pl in process:
                self._pipeline.append(pl)
        elif isinstance(process, SchedulerProcess):
            self._pipeline.append(process)

    def setPipelineJob(self, process: Union[SchedulerProcess, List[SchedulerProcess]]) -> None:
        """
        Remove all existed pipeline jobs and add the new job in to the job list
        """
        del self._pipeline
        self._pipeline = []
        if isinstance(process, List):
            for pl in process:
                self._pipeline.append(pl)
        elif isinstance(process, SchedulerProcess):
            self._pipeline.append(process)

    def __add__(self, other: Union[SchedulerProcess, List[SchedulerProcess], 'SchedulerPipeline']) \
            -> 'SchedulerPipeline':
        """ Add  """
        if isinstance(other, SchedulerProcess):
            # Add a Pipeline job in to the list
            self.addPipelineJob(other)
            return self
        elif isinstance(other, SchedulerPipeline):
            for job in other.pipeline:
                self.addPipelineJob(copy.deepcopy(job))
            return self


def findMaxT(layer: list) -> list:
    """
    Find maximum time for each layer

    :param layer: circuit list
    :return: a list containing the maximum time in each layer
    """
    maxDepth = [0 for _ in range(len(layer))]
    for layIdx, lay in enumerate(layer):
        for pair in lay:
            if pair is not None and pair.t > maxDepth[layIdx]:
                maxDepth[layIdx] = pair.t
    return maxDepth


def toLayer(layer: List[List[Optional[GatePulsePair]]], scheduler: 'Scheduler', reversedOrder: bool = False) -> None:
    """
    Turn GatePulsePair instances in a Scheduler object into layers

    :param layer: empty list to store GatePulsePair
    :param scheduler: a Scheduler object
    :param reversedOrder: transform by reversed order
    :return: None
    """
    if reversedOrder:
        for pair in reversed(scheduler.gatePulsePairs):
            if len(pair.qubits) == 1:
                layer[pair.qubits[0]].insert(0, pair)
            elif len(pair.qubits) >= 2:
                fillUp(layer, pair.qubits, reversedOrder)
                for qubit in pair.qubits:
                    layer[qubit].insert(0, pair)
    else:
        for pair in scheduler.gatePulsePairs:
            if isinstance(pair.qubits, List):
                if len(pair.qubits) == 1:
                    layer[pair.qubits[0]].append(pair)
                elif len(pair.qubits) >= 2:
                    fillUp(layer, pair.qubits)
                    for qubit in pair.qubits:
                        layer[qubit].append(pair)
            elif isinstance(pair.qubits, int):
                layer[pair.qubits].append(pair)


def fillUp(layer: List[List[Optional[GatePulsePair]]], qubits: List = None, reversedOrder: bool = False) -> None:
    """
    Fill up till each list is of equal size

    :param layer: GatePulsePair list
    :param qubits: qubit indexes to fill
    :param reversedOrder: transform by reversed order
    :return: None
    """
    if qubits is None:
        qubits = range(len(layer))
    lens = [len(layer[qubit]) for qubit in qubits]

    for qubit in qubits:
        if max(lens) - len(layer[qubit]) != 0:
            N = [None] * (max(lens) - len(layer[qubit]))
            if reversedOrder:
                layer[qubit] = N + layer[qubit]
            else:
                layer[qubit] = layer[qubit] + N


def addToJob(layer: List[List[Optional[GatePulsePair]]], scheduler: 'Scheduler', maxT: list) -> None:
    """
    Add the circuit in the given layer to scheduler.job

    :param layer: GatePulsePair list
    :param scheduler: original Scheduler object
    :param maxT: a list generated by findMaxt()
    """

    t0 = 0
    padding = scheduler.padding

    # For each layer
    for _layerIdx in range(len(layer)):
        # For each qubit
        for _pair in layer[_layerIdx]:
            if _pair is not None:
                if t0 == 0:
                    _pair.t0 = t0
                else:
                    if (t0 - padding * 2) < 0:
                        raise ValueError("Error: buffer time exceeds pulse length.")
                    _pair.t0 = t0 - padding * (_layerIdx + 1)
        if t0 == 0:
            t0 += maxT[_layerIdx] - padding
        else:
            t0 += maxT[_layerIdx] - padding * 2


def leftAlignSingleGates(layer: List[List[Optional[GatePulsePair]]]):
    """
    Left align all the single-qubit gates

    :param layer: GatePulsePair list
    """

    for qubit in layer:
        index = len(qubit) - 1
        while index >= 0:
            if qubit[index] is not None and len(qubit[index].qubits) == 1 \
                    and index - 1 >= 0 and qubit[index - 1] is None:
                qubit[index - 1], qubit[index] = qubit[index], qubit[index - 1]
                if index + 2 < len(qubit):
                    index += 2
            index -= 1
    return layer


def delayFirstGate(layer: List[List[Optional[GatePulsePair]]], gateNumberBeforeMulti):
    """
    Delay the first gates
    """

    for i in range(len(layer[0])):
        tempList = []
        for j in range(gateNumberBeforeMulti[i]):
            if layer[j][i] is not None:
                tempList.append(layer[j][i])
        if len(tempList) != gateNumberBeforeMulti[i]:
            count = len(tempList) - 1
            for num in reversed(range(gateNumberBeforeMulti[i])):
                if count >= 0:
                    layer[num][i] = tempList[count]
                else:
                    layer[num][i] = None
                count -= 1
