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
Readout simulator Template: 3-qubit readout simulator.
"""

from Quanlse.Superconduct.Simulator import ReadoutModel
from Quanlse.Superconduct.Simulator.PulseSim3Q import pulseSim3Q
from typing import Union
from math import pi


def readoutSim3Q(dt: Union[int, float] = 20.) -> ReadoutModel:
    r"""
    Return a template of 3-qubit readout model.

    :param dt: the sampling time period.

    :return: a readout simulator ReadoutModel object.
    """

    dt = dt  # The sampling time
    kappa = 0.0020 * (2 * pi)  # the system decay rate to the environment
    gain = 1000000.
    pulseModel = pulseSim3Q(dt=0.2)

    # resonator level
    resonatorLevel = 3  # resonator level
    resonatorFreq = {
        0: 7.104 * (2 * pi),
        1: 7.014 * (2 * pi),
        2: 7.214 * (2 * pi)
    }

    # qubit-resonator coupling strength
    coupling = {
        0: 0.134 * (2 * pi),
        1: 0.112 * (2 * pi),
        2: 0.121 * (2 * pi)
    }

    model = ReadoutModel(pulseModel=pulseModel, resonatorFreq=resonatorFreq, level=resonatorLevel,
                         coupling=coupling, dissipation=kappa, gain=gain, dt=dt)

    return model

