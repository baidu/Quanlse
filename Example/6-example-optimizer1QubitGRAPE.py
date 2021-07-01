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
Example: Single-qubit Gradient Ascent Pulse Engineering (GRAPE)
Please visit https://quanlse.baidu.com/#/doc/tutorial-GRAPEoptimizer for more details about this example.
"""

from math import pi

from Quanlse.QOperator import duff
from Quanlse.QHamiltonian import QHamiltonian as QHam
from Quanlse.QOperation import FixedGate
from Quanlse import Define
from Quanlse.remoteOptimizer import remoteOptimize1QubitGRAPE as runOptimize1QubitGRAPE


# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# Gate duration time.
tg = 40

# Number of qubit(s).
qubits = 1

# System energy level.
level = 2

# Sampling period.
dt = 1.0

# Anharmonicity of the qubit.
alphaq = - 0.22 * (2 * pi)

# ---------------------------------
# Construct the system Hamiltonian.
# ---------------------------------

ham = QHam(qubits, level, dt)
ham.addDrift(duff(level), 0, alphaq)

# --------------------------
# Define the input parameters.
# --------------------------

# The target gate.
ugoal = FixedGate.X.getMatrix()

# The channel(s) to be turned on.
xyzPulses = [1, 0, 0]

# The maximum number of iteration.
iterate = 40

# ------------------------------------------
# Run the optimization and show the results.
# ------------------------------------------

job, infid = runOptimize1QubitGRAPE(ham, ugoal, iterate=iterate, tg=tg, xyzPulses=xyzPulses)
print(infid)

# Plot the waveform.
job.plot()
