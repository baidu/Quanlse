#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2020 Baidu, Inc. All Rights Reserved.
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
Simulator Template: 3-qubit pulse simulator.
"""

import copy
from numpy import pi

from Quanlse.Simulator import PulseModel
from Quanlse.QHamiltonian import QHamiltonian
from Quanlse.QOperator import driveZ
from Quanlse.QWaveform import QJob, quasiSquareErf
from Quanlse.Scheduler.Superconduct.DefaultPulseGenerator import generate1Q
from Quanlse.Scheduler.Superconduct import SchedulerPulseGenerator


def pulseGenerator(ham: QHamiltonian) -> SchedulerPulseGenerator:
    r"""
    The pulseGenerator for the simulator.

    :param ham: a QHamiltonian object.
    :return: a SchedulerPulseGenerator object.
    """
    generator = SchedulerPulseGenerator(ham)

    # Add the generator for single qubit gates
    gateList1q = ['I', 'X', 'Y', 'Z', 'H', 'S', 'T', 'RX', 'RY', 'RZ', 'W', 'SQRTW', 'U']
    generator.addGenerator(gateList1q, generate1Q)

    # Add the generator for two qubit gates
    def generateCzSim(ham: QHamiltonian, cirLine, scheduler) -> QJob:
        """ Generate CZ gate """
        subHam = copy.deepcopy(ham)
        qIndex = tuple(cirLine.qRegIndexList)
        tg, x = scheduler.conf["caliDataCZ"][qIndex][0], scheduler.conf["caliDataCZ"][qIndex][1:]
        startId = 0
        qi1, qi2 = startId, startId + 1
        qj1, qj2 = startId + 2, startId + 3
        subHam.addWave(driveZ(ham.sysLevel), qIndex[0],
                       quasiSquareErf(0, tg, x[qi1], tg * x[qi2], tg * (1 - x[qi2]), 0.3 * x[qi1]))
        subHam.addWave(driveZ(ham.sysLevel), qIndex[1],
                       quasiSquareErf(0, tg, x[qj1], tg * x[qj2], tg * (1 - x[qj2]), 0.3 * x[qj1]))

        return subHam.job

    generator.addGenerator(['CZ'], generateCzSim)

    return generator


def pulseSim3Q(dt=0.5) -> PulseModel:
    r"""
    Return a template of 3-qubit simulator.

    :param dt: a sampling time period.
    :return: a 3-qubit PulseModel object.
    """

    # Define parameters needed
    dt = dt  # The sampling time
    sysLevel = 3  # The system level
    qubitNum = 3  # The number of qubits

    # Coupling map
    couplingMap = {
        (0, 1): 0.0380 * (2 * pi),
        (1, 2): 0.0076 * (2 * pi)
    }

    # Qubits frequency anharmonicity
    anharm = - 0.33 * (2 * pi)
    qubitAnharm = {0: anharm, 1: anharm, 2: anharm}  # The anharmonicities for each qubit

    # Qubit Frequency
    qubitFreq = {
        0: 5.5904 * (2 * pi),
        1: 4.7354 * (2 * pi),
        2: 4.8524 * (2 * pi)
    }

    model = PulseModel(subSysNum=qubitNum, sysLevel=sysLevel, couplingMap=couplingMap,
                       qubitFreq=qubitFreq, dt=dt, qubitAnharm=qubitAnharm, pulseGenerator=pulseGenerator,
                       )

    # Two-Qubit gate Calibration data
    model.conf["caliDataCZ"] = {
        (0, 1): [35.0, -3.5627, 0.1457, 0.1141, 0.45],
        (1, 2): [70.0, -1.1882, 0.1642, 0.1376, 0.1791]
    }

    return model
