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
Default Pulse Generator for QuanlseSchedulerSuperconduct.
"""

from Quanlse.Scheduler import SchedulerPulseGenerator
from Quanlse.QOperation import CircuitLine, Error
from Quanlse.QWaveform import QJob

from Quanlse.remoteOptimizer import remoteOptimize1Qubit as opt1q
from Quanlse.remoteOptimizer import remoteOptimizeCr as optCr
from Quanlse.remoteOptimizer import remoteOptimizeCz as optCz
from Quanlse.remoteOptimizer import remoteOptimizeISWAP as optISWAP


def generate1Q(ham: 'QHamiltonian' = None, cirLine: CircuitLine = None,
               scheduler: 'SchedulerSuperconduct' = None) -> QJob:
    """
    Default generator for single qubit gates.

    :param ham: QHam object containing the system information
    :param cirLine: a CircuitLine object containing the gate information
    :param scheduler: the instance of Quanlse Scheduler Superconducting
    :return: returned QJob object
    """
    subHam = ham.subSystem(cirLine.qRegIndexList)
    if cirLine.data.name in ['X', 'RX']:
        job, inf = opt1q(subHam, cirLine.data.getMatrix(), depth=2, targetInfid=0.0001)
    elif cirLine.data.name in ['Y', 'RY']:
        job, inf = opt1q(subHam, cirLine.data.getMatrix(), depth=2, targetInfid=0.0001)
    elif cirLine.data.name in ['Z', 'S', 'T', 'RZ']:
        job, inf = opt1q(subHam, cirLine.data.getMatrix(), depth=4, targetInfid=0.0001)
    elif cirLine.data.name in ['U', "H"]:
        job, inf = opt1q(subHam, cirLine.data.getMatrix(), depth=4, targetInfid=0.0001)
    else:
        raise Error.ArgumentError(f"Unsupported single qubit gate {cirLine.data.name}.")

    print(f"Infidelity of {cirLine.data.name} on qubit {cirLine.qRegIndexList}: {inf}")

    subHam.job = job
    return subHam.outputInverseJob(ham.subSysNum, ham.sysLevel, ham.dt)


def generateCr(ham: 'QHamiltonian' = None, cirLine: CircuitLine = None,
               scheduler: 'SchedulerSuperconduct' = None) -> QJob:
    """
    Default generator for Cross-resonance gate.

    :param ham: QHam object containing the system information
    :param cirLine: a CircuitLine object containing the gate information
    :param scheduler: the instance of Quanlse Scheduler Superconducting
    :return: returned QJob object
    """
    subHam = ham.subSystem(cirLine.qRegIndexList)
    job, inf = optCr(subHam, (-3.0, 3.0), maxIter=1)
    print(f"Infidelity of {cirLine.data.name} on qubit {cirLine.qRegIndexList}: {inf}")
    subHam.job = job
    return subHam.outputInverseJob(ham.subSysNum, ham.sysLevel, ham.dt)


def generateCz(ham: 'QHamiltonian' = None, cirLine: CircuitLine = None,
               scheduler: 'SchedulerSuperconduct' = None) -> QJob:
    """
    Default generator for controlled-Z gate.

    :param ham: QHam object containing the system information
    :param cirLine: a CircuitLine object containing the gate information
    :param scheduler: the instance of Quanlse Scheduler Superconducting
    :return: returned QJob object
    """
    subHam = ham.subSystem(cirLine.qRegIndexList)
    job, inf = optCz(subHam, tg=40, targetInfidelity=0.01)
    print(f"Infidelity of {cirLine.data.name} on qubit {cirLine.qRegIndexList}: {inf}")
    subHam.job = job
    return subHam.outputInverseJob(ham.subSysNum, ham.sysLevel, ham.dt)


def generateISWAP(ham: 'QHamiltonian' = None, cirLine: CircuitLine = None,
                  scheduler: 'SchedulerSuperconduct' = None) -> QJob:
    """
    Default generator for ISWAP gate.

    :param ham: QHam object containing the system information
    :param cirLine: a CircuitLine object containing the gate information
    :param scheduler: the instance of Quanlse Scheduler Superconducting
    :return: returned QJob object
    """
    subHam = ham.subSystem(cirLine.qRegIndexList)
    job, inf = optISWAP(subHam, tg=50, targetInfidelity=0.01)
    print(f"Infidelity of {cirLine.data.name} on qubit {cirLine.qRegIndexList}: {inf}")
    subHam.job = job
    return subHam.outputInverseJob(ham.subSysNum, ham.sysLevel, ham.dt)


def generatorCloud(ham: 'QHamiltonian') -> SchedulerPulseGenerator:
    """
    Return a default pulse SchedulerPulseGenerator instance for the scheduler.

    :param ham: a Hamiltonian object
    :return: returned generator object
    """

    generator = SchedulerPulseGenerator(ham)

    # Add the generator for single qubit gates
    gateList1q = ['X', 'Y', 'Z', 'H', 'S', 'T', 'RX', 'RY', 'RZ', 'W', 'SQRTW', 'U']
    generator.addGenerator(gateList1q, generate1Q)

    # Add the generator for two-qubit gates
    generator.addGenerator(['CR'], generateCr)
    generator.addGenerator(['CZ'], generateCz)
    generator.addGenerator(['ISWAP'], generateISWAP)

    return generator
