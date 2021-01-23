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

from enum import Enum, unique
from typing import Dict


@unique
class BackendName(Enum):
    """
    Name of Backends
    """

    

    CloudScheduler = 'cloud_quanlse_scheduler'
    """
    Cloud Simulator
    """


@unique
class Algorithm(Enum):
    """
    Algorithm
    """

    Normal = 'normal'
    """
    Normal
    """

    Fast = 'fast'
    """
    Fast
    """

    FastAverage = 'fast_average'
    """
    Fast Average
    """

    RungeKutta = 'runge_kutta'
    """
    Runge-Kutta Algorithm
    """


@unique
class HardwareImplementation(Enum):
    """
    Hardware implements
    """

    CZ = 'cz'
    CR = 'cr'


GateTimeDict = {
    'X': 40,
    'Y': 40,
    'Z': 40,
    'H': 40,
    'S': 40,
    'T': 40,
    'W': 40,
    'SQRTW': 40,
    'ISWAP': 40,
    'CZ': 40,
    'CR': 200,
    'CNOT': 200,
    'SWAP': 400,
    'U': 40,
    'RX': 40,
    'RY': 40,
    'RZ': 40,
}  # type: Dict[str,float]

TStepRange = (0.1, 10)
