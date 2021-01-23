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
Example: iSWAP gate optimizer
Please visit https://quanlse.baidu.com/#/doc/tutorial-iswap for more details about this example.
"""

from numpy import round
from math import pi

from Quanlse import Define
from Quanlse.Utils import Hamiltonian as qham
from Quanlse.Utils.Tools import project
from Quanlse.remoteOptimizer import remoteOptimizeISWAP as opt

from Quanlse.Utils import Operator

# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# Sampling period.
dt = 1.0

# Number of qubit(s).
qubits = 2

# System energy level.
level = 3

# --------------------------
# Define the qubit arguments
# --------------------------

qubitArgs = {
    "coupling": 0.0277 * (2 * pi),  # Coupling of Q0 and Q1
    "qubit_freq0": 5.805 * (2 * pi),  # Frequency of Q0
    "qubit_freq1": 5.205 * (2 * pi),  # Frequency of Q1
    "drive_freq0": 5.205 * (2 * pi),  # Drive frequency on Q0
    "drive_freq1": 5.205 * (2 * pi),  # Drive frequency on Q1
    "qubit_anharm0": -0.217 * (2 * pi),  # Anharmonicity of Q0
    "qubit_anharm1": -0.226 * (2 * pi)  # Anharmonicity of Q1
}

# ----------------------------
# Construct system Hamiltonian
# ----------------------------

# Create the Hamiltonian.
ham = qham.createHam(title="2q-3l", dt=dt, qubitNum=qubits, sysLevel=level)

for qu in range(2):
    # Add the detuning term(s).
    qham.addDrift(ham, name=f"q{qu}-detuning", onQubits=qu, matrices=Operator.number(level),
                  amp=(qubitArgs[f"qubit_freq{qu}"] - qubitArgs[f"drive_freq{qu}"]))

    # Add the anharmonicity term(s).
    qham.addDrift(ham, name=f"q{qu}-anharm", onQubits=qu, matrices=Operator.duff(level),
                  amp=qubitArgs[f"qubit_anharm{qu}"] / 2)

# Add the coupling term(s).
qham.addCoupling(ham, name="coupling", onQubits=[0, 1], g=qubitArgs["coupling"] / 2)

# Add the control term(s).
qham.addControl(ham, name="q0-ctrlz", onQubits=0, matrices=Operator.number(level))
qham.addControl(ham, name="q1-ctrlz", onQubits=1, matrices=Operator.number(level))

# ------------------------------------------
# Run the optimization and show the results.
# ------------------------------------------

# Set amplitude bound.
aBound = (-4, -3)

# Run the optimization.
ham, infidelity = opt(ham, aBound, tg=40, maxIter=5, targetInfidelity=0.01)

# Print infidelity and the waveforms.
print(f"minimum infidelity: {infidelity}")
qham.plotWaves(ham, ["q0-ctrlz", "q1-ctrlz"])

# Print the evolution process.
result = qham.simulate(ham)
projectedEvolution = project(result["unitary"], qubits, level, 2)
print("Projected evolution:\n", round(projectedEvolution, 2))

# Print the basic information of Hamiltonian.
qham.printHam(ham)
