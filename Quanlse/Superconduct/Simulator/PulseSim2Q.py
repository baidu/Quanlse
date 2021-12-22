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


def pulseSim2Q(dt: float = 0.5, frameMode: str = 'rot', qubitFreq: Dict[int, float] = None,
               qubitAnharm: Dict[int, float] = None, couplingMap: Dict[Tuple, Union[int, float]] = None) -> PulseModel:
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
        couplingMap = {
            (0, 1): 0.020 * (2 * pi),
        }
    else:
        pass

    # Qubits frequency anharmonicity
    if qubitAnharm is None:
        qubitAnharm = {
            0: - 0.22 * (2 * pi),
            1: - 0.22 * (2 * pi)
        }
    else:
        if len(qubitAnharm) != qubitNum:
            raise ArgumentError(f"The length of qubit anharmonicity should be {qubitNum}!")

    # Qubit Frequency
    if qubitFreq is None:
        qubitFreq = {
            0: 5.5004 * (2 * pi),
            1: 4.4546 * (2 * pi)
        }
    else:
        if len(qubitFreq) != qubitNum:
            raise ArgumentError(f"The length of qubit frequency should be {qubitNum}!")

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
    model = PulseModel(subSysNum=qubitNum, sysLevel=sysLevel, couplingMap=couplingMap, qubitFreq=qubitFreq,
                       driveFreq=driveFreq, dt=dt, qubitAnharm=qubitAnharm, pulseGenerator=pulseGenerator,
                       frameMode=frameMode)
    model.savePulse = False
    model.conf["frameMode"] = frameMode

    # Single-qubit gate calibration data
    model.conf["caliDataXY"] = {
        0: {'piAmp': 0.642781, 'piLen': 16.0, 'piTau': 8.0, 'piSigma': 2.0, 'dragCoef': -0.39289},
        1: {"piAmp": 0.649547, "piLen": 16.0, "piTau": 8.0, "piSigma": 2.0, "dragCoef": -0.347843}
    }

    # Flux pi pulse calibration data
    model.conf["caliDataZ"] = {
        0: {"piAmp": pi / 16., "piLen": 16.0},
        1: {"piAmp": pi / 16., "piLen": 16.0}
    }

    # Two-Qubit gate Calibration data
    model.conf["caliDataCZ"] = {
        (0, 1): {"czLen": 48.62, "q0ZAmp": -1.5506599, "q1ZAmp": 3.9979779,
                 "q0VZPhase": -0.079624 * 2 * pi, "q1VZPhase": 0.1592683 * 2 * pi}
    }

    return model
