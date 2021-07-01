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

from math import pi, sqrt

from Quanlse.QHamiltonian import QHamiltonian as QHam
from Quanlse.QOperator import driveX
from Quanlse.QWaveform import gaussian
from Quanlse.Utils.Infidelity import unitaryInfidelity
from Quanlse.QOperation.FixedGate import X


# Gate duration time.
tg = 20

# Number of qubit(s).
qubits = 1

# System energy level.
level = 2

# Sampling period.
dt = 0.2

# ---------------------------------
# Construct the system Hamiltonian.
# ---------------------------------

# Create the Hamiltonian with given parameters.
ham = QHam(subSysNum=qubits, sysLevel=level, dt=dt)

# Amplitude of the gaussian waveform with integral value of pi.
piAmp = pi / (tg / 8) / sqrt(2 * pi)

# Add gaussian waveform to control Hamiltonian.
ham.addWave(driveX(level), 0, gaussian(0, tg, piAmp, tg / 2, tg / 8))

# ------------------------------------------
# Run the simulation and show the results.
# ------------------------------------------

# Run the optimization.
results = ham.simulate()

# Show the result and plot the waveform.
print("Infidelity:", unitaryInfidelity(X.getMatrix(), results.result[0]["unitary"], 1))
ham.plot()

# Print the structure of a Hamiltonian.
print(ham)
