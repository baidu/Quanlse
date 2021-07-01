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
Simulator Template: 1-qubit pulse simulator.
"""

from numpy import pi

from Quanlse.Simulator import PulseModel
from Quanlse.QHamiltonian import QHamiltonian
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

    return generator


def pulseSim1Q(dt=0.5) -> PulseModel:
    r"""
    Return a template of 1-qubit simulator.

    :param dt: a sampling time period.
    :return: a 1-qubit PulseModel object.
    """

    # Define parameters needed
    dt = dt  # The sampling time
    sysLevel = 3  # The system level
    qubitNum = 1  # The number of qubits
    tRelax = 1000  # T1 relaxation time
    tDephase = 500  # T2 dephasing time

    # Qubits frequency anharmonicity
    anharm = - 0.34 * (2 * pi)
    qubitAnharm = {0: anharm}  # The anharmonicities for each qubit

    # Qubit Frequency
    qubitFreq = {0: 5.212 * (2 * pi)}

    # T1, T2
    t1 = {0: tRelax}
    t2 = {0: tDephase}

    model = PulseModel(subSysNum=qubitNum, sysLevel=sysLevel, T1=t1, T2=t2, qubitFreq=qubitFreq,
                       dt=dt, qubitAnharm=qubitAnharm, pulseGenerator=pulseGenerator)

    return model
