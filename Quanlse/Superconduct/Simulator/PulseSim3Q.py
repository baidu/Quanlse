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

from numpy import pi
from typing import Dict, Tuple, Union

from Quanlse.QPlatform.Error import ArgumentError
from Quanlse.Superconduct.Simulator import PulseModel
from Quanlse.Superconduct.SchedulerSupport.GeneratorPulseModel import pulseGenerator


def pulseSim3Q(dt: float = 0.5, frameMode: str = 'rot', qubitFreq: Dict[int, float] = None,
               qubitAnharm: Dict[int, float] = None, couplingMap: Dict[Tuple, Union[int, float]] = None) -> PulseModel:
    r"""
    Return a template of 3-qubit simulator.

    :param dt: a sampling time period.
    :param frameMode: indicates the frame, ``rot`` indicates the rotating frame,
        ``lab`` indicates the lab frame.
    :param qubitFreq: the qubit frequency, simulator will use the preset values if None.
    :param qubitAnharm: the qubit anharmonicity, simulator will use the preset values if None.
    :param couplingMap: the coupling between the qubits, simulator will use the preset values if None.
    :return: a 3-qubit PulseModel object.
    """

    # Define parameters needed
    sysLevel = 3  # The system level
    qubitNum = 3  # The number of qubits

    # Coupling map
    if couplingMap is None:
        couplingMap = {
            (0, 1): 0.020 * (2 * pi),
            (1, 2): 0.020 * (2 * pi)
        }
    else:
        pass

    # Qubits frequency anharmonicity
    if qubitAnharm is None:
        qubitAnharm = {
            0: - 0.22 * (2 * pi),
            1: - 0.22 * (2 * pi),
            2: - 0.22 * (2 * pi)
        }
    else:
        if len(qubitAnharm) != qubitNum:
            raise ArgumentError(f"The length of qubit anharmonicity should be {qubitNum}!")

    # Qubit Frequency
    if qubitFreq is None:
        qubitFreq = {
            0: 5.5004 * (2 * pi),
            1: 4.4546 * (2 * pi),
            2: 5.5004 * (2 * pi),
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

    # Single-Qubit gate Calibration data
    model.conf["caliDataXY"] = {
        0: {"piAmp": 0.642614, "piLen": 16.0, "piTau": 8.0, "piSigma": 2.0, "dragCoef": -0.36257},
        1: {"piAmp": 0.637912, "piLen": 16.0, "piTau": 8.0, "piSigma": 2.0, "dragCoef": -0.33307},
        2: {"piAmp": 0.642614, "piLen": 16.0, "piTau": 8.0, "piSigma": 2.0, "dragCoef": -0.36257}
    }

    # Flux pi pulse calibration data
    model.conf["caliDataZ"] = {
        0: {"piAmp": pi / 16., "piLen": 16.0},
        1: {"piAmp": pi / 16., "piLen": 16.0},
        2: {"piAmp": pi / 16., "piLen": 16.0}
    }

    # Two-Qubit gate Calibration data
    model.conf["caliDataCZ"] = {
        (0, 1): {"czLen": 35.00, "q0ZAmp": -3.5570, "q1ZAmp": 1.5022,
                 "q0VZPhase": -1.6924, "q1VZPhase": 1.0337},
        (1, 2): {"czLen": 47.24, "q0ZAmp": 2.0581, "q1ZAmp": -3.5250,
                 "q0VZPhase": 4.5778, "q1VZPhase": -3.9532},
    }

    return model
