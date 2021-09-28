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
Example: Cross-resonance gate optimizer
Please visit https://quanlse.baidu.com/#/doc/tutorial-cr for more details about this example.
"""

from numpy import round
from math import pi

from Quanlse.QHamiltonian import QHamiltonian as QHam
from Quanlse.Utils.Functions import project
from Quanlse.remoteOptimizer import remoteOptimizeCr as opt

from Quanlse.QOperator import duff, number
from Quanlse import Define


# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# Sampling period.
dt = 2.0

# Number of qubit(s).
qubits = 2

# System energy level.
level = 3

# ---------------------------
# Define the qubit arguments.
# ---------------------------

qubitArgs = {
    "coupling": 0.0038 * (2 * pi),  # Coupling of Q0 and Q1
    "qubit_freq0": 5.114 * (2 * pi),  # Frequency of Q0
    "qubit_freq1": 4.914 * (2 * pi),  # Frequency of Q1
    "drive_freq0": 4.914 * (2 * pi),  # Drive frequency on Q0
    "drive_freq1": 4.914 * (2 * pi),  # Drive frequency on Q1
    "qubit_anharm0": -0.33 * (2 * pi),  # Anharmonicity of Q0
    "qubit_anharm1": -0.33 * (2 * pi)  # Anharmonicity of Q1
}

# ----------------------------
# Construct system Hamiltonian.
# ----------------------------

# Create the Hamiltonian with given parameters.
ham = QHam(qubits, level, dt)
for qu in range(2):
    # Add the detuning terms.
    ham.addDrift(number, qu, (qubitArgs[f"qubit_freq{qu}"] - qubitArgs[f"drive_freq{qu}"]))

    # Add the anharmonicity terms.
    ham.addDrift(duff, qu, qubitArgs[f"qubit_anharm{qu}"] / 2)

# Add the coupling term.
ham.addCoupling([0, 1], qubitArgs["coupling"] / 2)

# ------------------------------------------
# Run the optimization and show the results.
# ------------------------------------------

# Set amplitude bound.
aBound = (-3.0, 3.0)

# Run the optimization.
gateJob, infidelity = opt(ham, aBound, tg=200, maxIter=5, targetInfidelity=0.01)

# Print the basic information of Hamiltonian.
print(ham)

# Print infidelity and the waveform(s).
print(f"Minimum infidelity: {infidelity}")
gateJob.plot()

# Print the evolution process.
result = ham.simulate(job=gateJob)
projectedEvolution = project(result.result[0]["unitary"], qubits, level, 2)
print("Projected evolution:\n", round(projectedEvolution, 2))
