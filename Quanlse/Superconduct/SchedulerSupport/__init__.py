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

r"""
Quanlse's Scheduler Superconduct achieves the following goals:

- It generates parameters and AWG input signal arrays for fidelity-optimized pulses
  when leakage into the state :math:`|2\rangle` is taken into account.

- It is also capable of scheduling pulses to minimize idle time and therefore reduce
  decoherence losses.

- At the same time, it allows for the visualization of pulse sequences for the users
  to check the results.

CNOT gate is rarely directly implemented on superconducting quantum chips. Instead,
it is often constructed by piecing together single-qubit gates and other two-qubit
gates like CR gate or ISWAP gate that can be easily implemented on a superconducting
chip (often called native gates). The two-qubit gates that are available in the
transmon-like superconducting qubit architecture can be divided into two categories:

**Flux-controlled**

This class of gates offers the advantage of short gate time to minimize decoherence
error. However, tuning the qubit frequency can introduce flux noises and lead to the
problem of frequency crowding.

**All-microwave control**

CR gate allows for an all-microwave control, which alleviates the problem of flux
noise. However, the much longer time-scale limits the gate fidelity (because of
the decoherence effects of qubits).

Since CZ gates can be used to construct a CNOT gate easily by using only two other
single-qubit gates, Quanlse's Scheduler offers this way to construct a CNOT gate
in a quantum circuit.
"""
import copy
from typing import List, Union

from Quanlse.Scheduler import Scheduler
from Quanlse.QHamiltonian import QHamiltonian as QHam
from Quanlse.Scheduler.SchedulerPulseGenerator import SchedulerPulseGenerator
from Quanlse.Scheduler.SchedulerPipeline import SchedulerPipeline


class SchedulerSuperconduct(Scheduler):
    """
    Basic class of Quanlse Scheduler for superconducting platform

    :param dt: AWG sampling time
    :param ham: the QHamiltonian object.
    :param generator: the pulseGenerator object.
    :param subSysNum: size of the subsystem
    :param sysLevel: the energy levels of the system (support different levels for different qubits)
    """
    def __init__(self, dt: float = None, ham: QHam = None, generator: SchedulerPulseGenerator = None,
                 pipeline: SchedulerPipeline = None, subSysNum: int = None,
                 sysLevel: Union[int, List[int], None] = None):
        """
        Constructor for SchedulerSuperconduct object
        """
        # Initialization
        super().__init__(dt, ham, generator, pipeline, subSysNum, sysLevel)

        # Add the pulse generator
        if self._pulseGenerator is None:
            if ham is not None:
                from Quanlse.Superconduct.SchedulerSupport.GeneratorCloud import generatorCloud
                self._pulseGenerator = generatorCloud(self._ham)
        else:
            self._pulseGenerator = copy.deepcopy(generator)

        # Add the pipeline
        if pipeline is not None:
            del self._pipeline
            self._pipeline = copy.deepcopy(pipeline)
        else:
            from Quanlse.Superconduct.SchedulerSupport.PipelineLeftAligned import leftAligned
            self.pipeline.addPipelineJob(leftAligned)

        # Set the local oscillator information
        self._subSysLO = []  # type: List[List[float]]
        for num in range(self.subSysNum):
            self._subSysLO.append([0.0, 0.0])
