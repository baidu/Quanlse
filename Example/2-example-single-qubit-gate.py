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
Example: Single-qubit gate optimizer
Please visit https://quanlse.baidu.com/#/doc/tutorial-single-qubit for more details about this example.
"""

from numpy import round
from math import pi

from Quanlse.QHamiltonian import QHamiltonian as QHam
from Quanlse.Utils.Functions import project
from Quanlse.QOperator import duff
from Quanlse.remoteOptimizer import remoteOptimize1Qubit as opt
from Quanlse.QOperation import FixedGate
from Quanlse import Define


# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# Number of qubit(s).
qubits = 1

# System energy level.
level = 3

# Sampling period.
dt = 0.2

# Anharmonicity of the qubit.
anharm = - 0.33 * (2 * pi)


# ---------------------------------
# Construct the system Hamiltonian.
# ---------------------------------

# Create the Hamiltonian.
ham = QHam(qubits, level, dt=dt)

# Add the drift term(s).
ham.addDrift(duff(level), 0, coef=anharm / 2)

# ------------------------------------------
# Run the optimization and show the results.
# ------------------------------------------

# Run the optimization.
gateJob, infidelity = opt(ham, FixedGate.Y.getMatrix(), depth=3, targetInfid=0.001)

# Print infidelity and the waveform(s).
print(f"minimum infidelity: {infidelity}")
gateJob.plot()

# Print the evolution process.
result = ham.simulate(job=gateJob)
projectedEvolution = project(result.result[0]["unitary"], qubits, level, 2)
print("Projected evolution:\n", round(projectedEvolution, 2))
