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
Simulator Template: 2-qubit pulse simulator.
"""

from numpy import pi
from typing import Dict, Tuple, Union

from Quanlse.QPlatform.Error import ArgumentError
from Quanlse.Superconduct.Simulator import PulseModel
from Quanlse.Superconduct.SchedulerSupport.GeneratorPulseModel import pulseGenerator
from Quanlse.Superconduct.Simulator.Utils import SimulatorLabSpace, SimulatorRunner, SimulatorGenerator
from Quanlse.Superconduct.Simulator.SimulatorAgent import SimulatorAgent
from Quanlse.QWaveform import QJob, QWaveform
from Quanlse.QOperator import chZ, chX, chY, driveX, driveY, driveZ


def simulator2Q(dt: float = 0.5, frameMode: str = 'rot', qubitFreq: Dict[int, float] = None,
                qubitAnharm: Dict[int, float] = None, couplingMap: Dict[Tuple, Union[int, float]] = None) \
        -> SimulatorAgent:
    r"""
    Return a template of 3-qubit simulator.

    :param dt: a sampling time period.
    :param frameMode: indicates the frame, ``rot`` indicates the rotating frame,
        ``lab`` indicates the lab frame.
    :param qubitFreq: the qubit frequency, simulator will use the preset values if None.
    :param qubitAnharm: the qubit anharmonicity, simulator will use the preset values if None.
    :param couplingMap: the coupling between the qubits, simulator will use the preset values if None.
    :return: a 2-qubit PulseModel object.
    """

    # Define parameters needed
    sysLevel = 3  # The system level
    qubitNum = 2  # The number of qubits

    # Coupling map
    if couplingMap is None:
        couplingMap = {(0, 1): 0.020 * (2 * pi)}

    # Qubits frequency anharmonicity
    if qubitAnharm is None:
        qubitAnharm = {0: - 0.22 * (2 * pi), 1: - 0.22 * (2 * pi)}
    else:
        if len(qubitAnharm) != qubitNum:
            raise ArgumentError(f"The length of qubit anharmonicity should be {qubitNum}!")

    # Qubit Frequency
    if qubitFreq is None:
        qubitFreq = {0: 5.5004 * (2 * pi), 1: 4.4546 * (2 * pi)}
    else:
        if len(qubitFreq) != qubitNum:
            raise ArgumentError(f"The length of qubit frequency should be {qubitNum}!")

    # Drive Frequency
    if frameMode == 'lab':
        driveFreq = qubitFreq
    elif frameMode == 'rot':
        driveFreq = None
    else:
        raise ArgumentError("Only rotating and lab frames are supported!")

    # Create PulseModel Instance
    model = PulseModel(subSysNum=qubitNum, sysLevel=sysLevel, couplingMap=couplingMap, qubitFreq=qubitFreq,
                       driveFreq=driveFreq, dt=dt, qubitAnharm=qubitAnharm, pulseGenerator=pulseGenerator,
                       frameMode='rot')
    model.savePulse = False
    model.conf["frameMode"] = frameMode

    # Define the waveform template
    gaussian = QWaveform(f="gaussian", args={"a": 0., "tau": 16., "sigma": 4.}, t=32., t0=0.)
    drag = QWaveform(f="drag", args={"a": 0., "tau": 16., "sigma": 4.}, t=32., t0=0.)
    squareWave = QWaveform(f="square", args={"a": 0.}, t=32., t0=0.)

    # Waveforms for X gate
    jobQ1X = QJob(subSysNum=qubitNum, dt=dt)
    gaussian.args["a"], drag.args["a"] = 0.3118644, -0.10769
    jobQ1X.addWave(chX, 0, waves=gaussian)
    jobQ1X.addWave(chY, 0, waves=drag)

    jobQ2X = QJob(subSysNum=qubitNum, dt=dt)
    gaussian.args["a"], drag.args["a"] = 0.3118644, -0.10769
    jobQ2X.addWave(chX, 1, waves=gaussian)
    jobQ2X.addWave(chY, 1, waves=drag)

    jobQ1HalfX = QJob(subSysNum=qubitNum, dt=dt)
    gaussian.args["a"], drag.args["a"] = 0.3118644 / 2, -0.10769 / 2
    jobQ1HalfX.addWave(chX, 0, waves=gaussian)
    jobQ1HalfX.addWave(chY, 0, waves=drag)

    jobQ2HalfX = QJob(subSysNum=qubitNum, dt=dt)
    gaussian.args["a"], drag.args["a"] = 0.3118644 / 2, -0.10769 / 2
    jobQ2HalfX.addWave(chX, 1, waves=gaussian)
    jobQ2HalfX.addWave(chY, 1, waves=drag)

    # Waveforms for Y gate
    jobQ1Y = QJob(subSysNum=qubitNum, dt=dt)
    gaussian.args["a"], drag.args["a"] = 0.3118644, 0.10769
    jobQ1Y.addWave(chY, 0, waves=gaussian)
    jobQ1Y.addWave(chX, 0, waves=drag)

    jobQ2Y = QJob(subSysNum=qubitNum, dt=dt)
    gaussian.args["a"], drag.args["a"] = 0.3118644, 0.10769
    jobQ2Y.addWave(chY, 1, waves=gaussian)
    jobQ2Y.addWave(chX, 1, waves=drag)

    jobQ1HalfY = QJob(subSysNum=qubitNum, dt=dt)
    gaussian.args["a"], drag.args["a"] = 0.3118644 / 2, +0.10769 / 2
    jobQ1HalfY.addWave(chY, 0, waves=gaussian)
    jobQ1HalfY.addWave(chX, 0, waves=drag)

    jobQ2HalfY = QJob(subSysNum=qubitNum, dt=dt)
    gaussian.args["a"], drag.args["a"] = 0.3118644 / 2, +0.10769 / 2
    jobQ2HalfY.addWave(chY, 1, waves=gaussian)
    jobQ2HalfY.addWave(chX, 1, waves=drag)

    # Waveforms for Z gate on the second qubit
    jobQ1Z = QJob(subSysNum=qubitNum, dt=dt)
    squareWave.args["a"] = pi / 16.
    jobQ1Z.addWave(chZ, 0, waves=squareWave)

    jobQ2Z = QJob(subSysNum=qubitNum, dt=dt)
    squareWave.args["a"] = pi / 16.
    jobQ2Z.addWave(chZ, 1, waves=squareWave)

    # Waveforms for CZ gate on the first-second qubit
    jobQ0Q1CZ = QJob(subSysNum=qubitNum, dt=dt)
    jobQ0Q1CZ.addWave(chZ, 0, QWaveform(f="square", t=40., args={"a": -2.434482}))
    jobQ0Q1CZ.addWave(chZ, 1, QWaveform(f="square", t=40., args={"a": 2.184210}))
    jobQ0Q1CZ.addWave(chZ, 0, QWaveform(f="square", args={"a": -0.080952}, t=30., t0=40.))
    jobQ0Q1CZ.addWave(chZ, 1, QWaveform(f="square", args={"a": 0.234920}, t=30., t0=40.))

    # Create LabSpace and initial parameters
    qubits = {0: "q1", 1: "q2"}
    channels = {
        "q1": [chX.name, chY.name, chZ.name],
        "q2": [chX.name, chY.name, chZ.name]
    }
    configDict = {
        # Basic parameters
        "basic.device": "sim-2q",
        "basic.description": "This is a simulator with two coupled qubits.",
        "basic.awg_dt": dt,
        "basic.sub_system_number": qubitNum,
        "basic.qubits": qubits,
        "basic.channels": channels,
        # Gates parameters for qubit 0
        "gates.q1.X": jobQ1X.dump2Json(),
        "gates.q1.X/2": jobQ1HalfX.dump2Json(),
        "gates.q1.Y": jobQ1Y.dump2Json(),
        "gates.q1.Y/2": jobQ1HalfY.dump2Json(),
        "gates.q1.Z": jobQ1Z.dump2Json(),
        # Gates parameters for qubit 0
        "gates.q2.X": jobQ2X.dump2Json(),
        "gates.q2.X/2": jobQ2HalfX.dump2Json(),
        "gates.q2.Y": jobQ2Y.dump2Json(),
        "gates.q2.Y/2": jobQ2HalfY.dump2Json(),
        "gates.q2.Z": jobQ2Z.dump2Json(),
        # Gates parameters for cz gate on q0 and q1
        "gates.q1q2.CZ": jobQ0Q1CZ.dump2Json(),
    }
    labSpace = SimulatorLabSpace(configDict)

    # Create the runner
    runner = SimulatorRunner(model)
    runner.setChannelOperator(chX.name, driveX(sysLevel))
    runner.setChannelOperator(chY.name, driveY(sysLevel))
    runner.setChannelOperator(chZ.name, driveZ(sysLevel))

    # Create Generator
    generator = SimulatorGenerator(labSpace=labSpace)

    # Create an agent instance
    agent = SimulatorAgent(model=model, labSpace=labSpace, runner=runner, generator=generator)
    return agent
