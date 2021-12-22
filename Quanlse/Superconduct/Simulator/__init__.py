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
Simulating the time evolution of the qubits at the pulse level gives us more insight into how
quantum gates operates and the effects of noise. For superconducting quantum circuit, the
transmon qubits are controlled by applying microwave and magnetic flux. However, the performance
of quantum gates are often suppressed by various factors - the decoherence of the qubit due
to its interaction with the environment, the unwanted cross-talk effect and leakage into the
higher levels of the transmon.

The pulse-level simulator provided by Quanlse allows us to simulate quantum operations on
noisy quantum device consisting of multiple transmon qubits to better understand the physics
behind quantum computing.

For more details, please visit https://quanlse.baidu.com/#/doc/tutorial-pulse-level-simulator.
"""

from .PulseModel import PulseModel, ReadoutModel, ReadoutPulse
from .PulseSim1Q import pulseSim1Q
from .PulseSim2Q import pulseSim2Q
from .PulseSim3Q import pulseSim3Q
from .PulseSimQCQ import pulseSimQCQ
from .ReadoutSim3Q import readoutSim3Q
from .SimulatorAgent import SimulatorAgent
