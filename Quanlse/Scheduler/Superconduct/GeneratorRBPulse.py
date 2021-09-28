#!/usr/bin/python3
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
Pulse Generator for QuanlseSchedulerSuperconduct (Randomized Benchmarking).
"""

from Quanlse.Scheduler.Superconduct import GeneratorCloud
from Quanlse.QOperation import CircuitLine
from Quanlse.QWaveform import QJob
from Quanlse.QHamiltonian import QHamiltonian as QHam
from Quanlse.Scheduler.SchedulerPulseGenerator import SchedulerPulseGenerator

from Quanlse.remoteOptimizer import remoteOptimize1Qubit as opt1q


def generate1QClifford(ham: QHam = None, cirLine: CircuitLine = None, scheduler: 'SchedulerSuperconduct' = None)\
        -> QJob:
    """
    Default generator for single qubit gates.

    :param ham: QHam object containing the system information
    :param cirLine: a CircuitLine object containing the gate information
    :param scheduler: the instance of Quanlse Scheduler Superconducting
    :return: returned QJob object
    """
    subHam = ham.subSystem(cirLine.qRegIndexList)

    # Use the pulses we generate based on the qubit chosen
    if cirLine.data.name in ['Cinv']:
        job, inf = opt1q(subHam, cirLine.data.getMatrix(), depth=6, targetInfid=0.0002)
    else:
        job, inf = opt1q(ham, cirLine.data.getMatrix(), depth=4, targetInfid=0.0001)

    print(f"Infidelity of {cirLine.data.name} on qubit {cirLine.qRegIndexList}: {inf}")

    subHam.job = job
    return subHam.outputInverseJob(ham.subSysNum, ham.sysLevel, ham.dt)


def SingleQubitCliffordPulseGenerator(ham: QHam) -> SchedulerPulseGenerator:
    """
    Return a single-qubit Clifford pulse SchedulerPulseGenerator instance for the scheduler.

    :param ham: a Hamiltonian object
    :return: returned generator object
    """
    generator = SchedulerPulseGenerator(ham)
    gateList1q = ['X', 'Y', 'Z', 'H', 'S', 'T', 'RX', 'RY', 'RZ', 'W', 'SQRTW', 'U']
    generator.addGenerator(gateList1q, GeneratorCloud.generate1Q)
    gateList1QCliff = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15',
                       'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'Cinv']
    generator.addGenerator(gateList1QCliff, generate1QClifford)
    generator.addGenerator(['CR'], GeneratorCloud.generateCr)
    generator.addGenerator(['CZ'], GeneratorCloud.generateCz)
    generator.addGenerator(['ISWAP'], GeneratorCloud.generateISWAP)
    return generator
