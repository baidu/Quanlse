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
Default Pulse Generator for QuanlseIon.
"""

from math import pi
from numpy import arccos

from Quanlse.QOperation import CircuitLine, Error
from Quanlse.QWaveform import QJob
from Quanlse.QWaveform import QWaveform
from Quanlse.QOperator import QOperator
from Quanlse.Scheduler.SchedulerPulseGenerator import SchedulerPulseGenerator
from Quanlse.remoteOptimizer import remoteIonOptimize1Qubit as opt1q
from Quanlse.remoteOptimizer import remoteIonGeneralMS as optGMS


uWave = QOperator('qubit', None)


def generateIon1Qubit(cirLine: CircuitLine = None, scheduler: 'SchedulerIon' = None) -> QJob:
    """
    Default generator for ion single-qubit gates.

    :param cirLine: input circuit line object
    :param scheduler: input scheduler object
    :return: a returned QJob object
    """
    job = QJob(subSysNum=scheduler.subSysNum, sysLevel=2, dt=scheduler.dt)
    qLabel = cirLine.qRegIndexList
    matrixNumber = cirLine.data.getMatrix()
    theta = arccos(matrixNumber[0][0])
    theta = theta.real
    if cirLine.data.name == 'X':
        amp, inf, ureal = opt1q("ionRx", pi, scheduler.dt)
        print(f"Single gate infidelity on qubit {qLabel[0]} is: \n {inf}")
        job.addWave(uWave, onSubSys=qLabel[0], waves=QWaveform(t0=0.0, t=2 * scheduler.dt, seq=[amp, 0]))
    elif cirLine.data.name == 'Y':
        amp, inf, ureal = opt1q("ionRy", pi, scheduler.dt)
        print(f"Single gate infidelity on qubit {qLabel[0]} is: \n {inf}")
        job.addWave(uWave, onSubSys=qLabel[0], waves=QWaveform(t0=0.0, t=2 * scheduler.dt, seq=[amp, 0]))
    elif cirLine.data.name == 'RX':
        amp, inf, ureal = opt1q("ionRx", theta, scheduler.dt)
        print(f"Single gate infidelity on qubit {qLabel[0]} is: \n {inf}")
        job.addWave(uWave, onSubSys=qLabel[0], waves=QWaveform(t0=0.0, t=2 * scheduler.dt, seq=[amp, 0]))
    elif cirLine.data.name == 'RY':
        amp, inf, ureal = opt1q("ionRy", theta, scheduler.dt)
        print(f"Single gate infidelity on qubit {qLabel[0]} is: \n {inf}")
        job.addWave(uWave, onSubSys=qLabel[0], waves=QWaveform(t0=0.0, t=2 * scheduler.dt, seq=[amp, 0]))
    else:
        raise Error.ArgumentError(f"Unsupported single-qubit gate {cirLine.data.name}.")

    return job


def generateMS(cirLine: CircuitLine = None, scheduler: 'SchedulerIon' = None) -> QJob:
    """
    Default generator for M-S gate.

    :param cirLine: input circuit line object
    :param scheduler: input scheduler object
    :return: a returned QJob object
    """
    if 5 < scheduler.dt < 15:
        job = QJob(subSysNum=scheduler.subSysNum, sysLevel=2, dt=scheduler.dt)
        qLabel = cirLine.qRegIndexList
        gatePair = [[qLabel[0], qLabel[1]]]

        """
        For parameter args1, scheduler.subSysNum is the total ion in chain.
        And simply, we fix the trapped potential with some typical value:
        omegaXY=2 * pi * 2e6, omegaZ=2 * pi * 0.2e6. 
        We choose typical atomMass=171 for Ytterbium atom, 
        and using "transverse" phonon to transmit information.
        """
        args1 = (scheduler.subSysNum, 171, 2 * pi * 2e6, 2 * pi * 0.2e6, "transverse")

        """
        And for quantum gate parameter, we fix the square pulse number
        to be 3*N, where N=scheduler.subSysNum is the ion number in chain.
        The laser detuning mu usually has relation with gate time 
        mu=2*pi/tg, where tg=scheduler.subSysNum * 3 * scheduler.dt.
        """
        args2 = (scheduler.subSysNum * 3, scheduler.dt, 2 * pi / (scheduler.subSysNum * 3 * scheduler.dt))

        result, unitary = optGMS(gatePair, args1=args1, args2=args2)
        infidelity = result['infidelity']
        print(f"Gate infidelity of MS{gatePair[0]} is: \n {infidelity}")
        sequence = result['pulse_list'][0]

        job.addWave(uWave, onSubSys=qLabel[0], waves=QWaveform(t0=0.0, seq=sequence))
        job.addWave(uWave, onSubSys=qLabel[1], waves=QWaveform(t0=0.0, seq=sequence))
    else:
        raise Error.ArgumentError("The sample time of MS gate is between 5 - 15.")

    return job


def defaultPulseGenerator() -> SchedulerPulseGenerator:
    """
    Default pulse generator for ion-trap qubit

    :return: returned generator object
    """
    generator = SchedulerPulseGenerator()

    # Add the generator for single qubit gates
    gateList1q = ['X', 'Y', 'RX', 'RY']
    generator.addGenerator(gateList1q, generateIon1Qubit)

    # Add the generator for MS gates
    generator.addGenerator(['MS'], generateMS)

    return generator

