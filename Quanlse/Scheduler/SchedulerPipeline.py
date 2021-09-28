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
from typing import List, Union, Dict, Any, Callable

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

    def __call__(self, job: QJob, scheduler: 'Scheduler') -> QJob:
        """
        Call the scheduling function.
        """
        if self._processFunc is None:
            raise ArgumentError("Scheduling function is not set.")
        return self._processFunc(job=job, scheduler=scheduler)


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

    def __call__(self, job: QJob, scheduler: 'Scheduler') -> QJob:
        """
        Call the instance and run the pipeline in order,
        and output a QJob instance.
        """
        for idx, process in enumerate(self._pipeline):
            if isinstance(process, Callable):
                job = process(job, scheduler)
            else:
                raise ArgumentError("Scheduler process is not callable.")
        return job

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


def findMaxT(layer: list, scheduler: 'Scheduler') -> list:
    """
    Find maximum time for each layer

    :param layer: circuit list
    :param scheduler: a Scheduler object
    :return: a list containing the maximum time in each layer
    """
    maxDepth = [0 for _ in range(len(layer))]
    for layIdx, lay in enumerate(layer):
        for gateIdx in range(len(lay)):
            if lay[gateIdx] is not None:
                if scheduler.savePulse:
                    someJob = scheduler.findGatePulsePair(lay[gateIdx])
                else:
                    job = scheduler.pulseGenerator(lay[gateIdx], scheduler)
                    someJob = GatePulsePair(lay[gateIdx], job)
                if someJob.t > maxDepth[layIdx]:
                    maxDepth[layIdx] = someJob.t
    return maxDepth


def toLayer(layer: list, scheduler: 'Scheduler') -> None:
    """
    Turn gates in a Scheduler object into layers

    :param layer: empty list to store circuit
    :param scheduler: a Scheduler object
    :return: None
    """
    for gate in scheduler.circuit:
        if len(gate.qRegIndexList) == 1:
            layer[gate.qRegIndexList[0]].append(gate)
        elif len(gate.qRegIndexList) >= 2:
            fillUp(layer, gate.qRegIndexList)
            for qubit in gate.qRegIndexList:
                layer[qubit].append(gate)


def fillUp(layer: list, qubits: list = None) -> None:
    """
    Fill up till each list is of equal size

    :param layer: circuit list
    :param qubits: qubit indexes to fill
    :return: None
    """
    if qubits is None:
        qubits = range(len(layer))

    lens = [len(layer[qubit]) for qubit in qubits]
    for qubit in qubits:
        if max(lens) - len(layer[qubit]) != 0:
            N = [None] * (max(lens) - len(layer[qubit]))
            layer[qubit] = layer[qubit] + N


def addToJob(layer: list, scheduler: 'Scheduler', maxT: list, job: QJob) -> None:
    """
    Add the circuit in the given layer to scheduler.job

    :param layer: circuit list
    :param scheduler: original Scheduler object
    :param maxT: a list generated by findMaxt()
    :param job: the QJob object to add the pulses to
    """

    t0 = 0
    padding = scheduler.padding

    # for each layer
    for i in range(len(layer)):
        temp = []

        # for each qubit
        for j in layer[i]:

            # dont add the same gate multiple times
            flag = False
            for tp in temp:
                if tp == j and j is not None:
                    flag = True
            if j is not None and flag is False:
                if scheduler.savePulse:
                    obj = scheduler.findGatePulsePair(j)
                else:
                    jobObj = scheduler.pulseGenerator(j, scheduler)
                    obj = GatePulsePair(j, jobObj)
                if t0 == 0:
                    job.appendJob(obj.pulses, t0)
                else:
                    if (t0 - padding * 2) < 0:
                        raise ValueError("Error: buffer time exceeds pulse length.")
                    job.appendJob(obj.pulses, t0 - padding * (i + 1))

            temp.append(j)

        if t0 == 0:
            t0 += maxT[i] - padding
        else:
            t0 += maxT[i] - padding * 2


def reverseToLayer(layer: list, scheduler: 'Scheduler'):
    """
    Turn gates in Scheduler into layers
    """
    for gate in reversed(scheduler.circuit):
        if len(gate.qRegIndexList) == 1:
            layer[gate.qRegIndexList[0]].insert(0, gate)
        elif len(gate.qRegIndexList) >= 2:
            reverseFillUp(layer, gate.qRegIndexList)
            for qubit in gate.qRegIndexList:
                layer[qubit].insert(0, gate)


def reverseFillUp(layer: list, qubits=None):
    """
    Fill up till the list is of equal size
    """

    if qubits is None:
        qubits = range(len(layer))

    lens = [len(layer[qubit]) for qubit in qubits]
    for qubit in qubits:
        if max(lens) - len(layer[qubit]) != 0:
            N = [None] * (max(lens) - len(layer[qubit]))
            layer[qubit] = N + layer[qubit]


def leftAlignSingleGates(layer: list):
    """
    Left align all the single-qubit gates
    """

    for qubit in layer:
        index = len(qubit) - 1
        while index >= 0:
            if qubit[index] is not None and len(qubit[index].qRegIndexList) == 1 \
                    and index - 1 >= 0 and qubit[index - 1] is None:
                qubit[index - 1], qubit[index] = qubit[index], qubit[index - 1]
                if index + 2 < len(qubit):
                    index += 2
            index -= 1
    return layer


def delayFirstGate(layer, gateNumberBeforeMulti):
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
