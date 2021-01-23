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

"""
Example: Pi pulse
Please visit https://quanlse.baidu.com/#/doc/tutorial-pi-pulse for more details about this example.
"""

from numpy import round
from math import pi, sqrt

from Quanlse.Utils import Hamiltonian as qham
from Quanlse.Utils import Operator

# Sampling period.
dt = 0.2

# Number of qubit(s).
qubits = 1

# System energy level.
level = 2

# Gate duration time.
tg = 20

# --------------------------------
# Construct the system Hamiltonian
# --------------------------------

# Create the Hamiltonian.
ham = qham.createHam(title="1q-2l", dt=dt, qubitNum=qubits, sysLevel=level)

# Add the control term(s).
qham.addControl(ham, name="q0-ctrlx", onQubits=0, matrices=Operator.driveX(level))

# Add Pi pulse wave(s).
# `amp` is calculated from the integral of Gaussian function.
amp = pi / (tg / 8) / sqrt(2 * pi)
qham.addWave(ham, "q0-ctrlx", f="gaussian", t0=0, t=tg, para={"a": amp, "tau": tg / 2, "sigma": tg / 8})

# Print the basic information of Hamiltonian.
qham.printHam(ham)

# Simulate the evolution and print the result.
unitary = qham.getUnitary(ham)
print("Evolution unitary:\n", round(unitary, 2))

# Print the waveform.
qham.plotWaves(ham)
