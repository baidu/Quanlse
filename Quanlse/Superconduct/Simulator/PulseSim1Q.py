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
from typing import Dict

from Quanlse.Superconduct.Simulator import PulseModel
from Quanlse.Superconduct.SchedulerSupport.GeneratorPulseModel import pulseGenerator
from Quanlse.QPlatform.Error import ArgumentError


def pulseSim1Q(dt: float = 0.5, frameMode: str = 'rot', qubitFreq: Dict[int, float] = None,
               qubitAnharm: Dict[int, float] = None) -> PulseModel:
    r"""
    Return a template of 1-qubit simulator.

    :param dt: a sampling time period.
    :param frameMode: indicates the frame, ``rot`` indicates the rotating frame,
        ``lab`` indicates the lab frame.
    :param qubitFreq: the qubit frequency, simulator will use the template values if None.
    :param qubitAnharm: the qubit anharmonicity, simulator will use the preset values if None.
    :return: a 1-qubit PulseModel object.
    """

    # Define parameters needed
    sysLevel = 3  # The system level
    qubitNum = 1  # The number of qubits
    tRelax = 1000  # T1 relaxation time
    tDephase = 500  # T2 dephasing time

    # Qubits frequency anharmonicity
    if qubitAnharm is None:
        qubitAnharm = {
            0: - 0.33 * (2 * pi)
        }
    else:
        if len(qubitAnharm) != qubitNum:
            raise ArgumentError(f"The length of qubit frequency should be {qubitNum}!")

    # Qubit Frequency
    if qubitFreq is None:
        qubitFreq = {0: 5.212 * (2 * pi)}
    else:
        if len(qubitFreq) != qubitNum:
            raise ArgumentError(f"The length of qubit frequency should be {qubitNum}!")

    # T1, T2
    t1 = {0: tRelax}
    t2 = {0: tDephase}

    # Drive Frequency
    if frameMode == 'lab':
        # Set the local oscillator
        driveFreq = qubitFreq
    elif frameMode == 'rot':
        driveFreq = None
        pass
    else:
        raise ArgumentError("Only rotating and lab frames are supported!")

    # Generate the pulse model
    model = PulseModel(subSysNum=qubitNum, sysLevel=sysLevel, T1=t1, T2=t2, qubitFreq=qubitFreq, driveFreq=driveFreq,
                       dt=dt, qubitAnharm=qubitAnharm, frameMode=frameMode, pulseGenerator=pulseGenerator)
    model.savePulse = False
    model.conf["frameMode"] = frameMode

    # Flux pi pulse calibration data
    model.conf["caliDataZ"] = {
        0: {"piAmp": pi / 16., "piLen": 16.0}
    }

    # Single-Qubit gate Calibration data
    model.conf["caliDataXY"] = {
        0: {"piAmp": 0.628965, "piLen": 16.0, "piTau": 8.0, "piSigma": 2.0, "dragCoef": - 0.23}
    }

    return model
