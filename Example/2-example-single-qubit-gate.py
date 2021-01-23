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
Example: Single qubit gate optimizer
Please visit https://quanlse.baidu.com/#/doc/tutorial-single-qubit for more details about this example.
"""

from numpy import round
from math import pi

from Quanlse import Define
from Quanlse.Utils import Hamiltonian as qham
from Quanlse.Utils.Tools import project
from Quanlse.remoteOptimizer import remoteOptimize1Qubit as opt

from Quanlse.QOperation import FixedGate
from Quanlse.Utils import Operator

# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# Sampling period.
dt = 1.0

# Number of qubit(s).
qubits = 1

# System energy level.
level = 3

# --------------------------
# Define the qubit arguments
# --------------------------

qubitArgs = {
    "qubit_anharm": -0.33 * (2 * pi)  # Anharmonicity of the qubit
}

# --------------------------------
# Construct the system Hamiltonian
# --------------------------------

# Create the Hamiltonian.
ham = qham.createHam(title="1q-3l", dt=dt, qubitNum=qubits, sysLevel=level)

# Add the anharmonicity term(s).
qham.addDrift(ham, name="q0-anharm", onQubits=0, matrices=Operator.duff(level), amp=qubitArgs["qubit_anharm"] / 2)

# Add the control terms.
qham.addControl(ham, name="q0-ctrlx", onQubits=0, matrices=Operator.driveX(level))
qham.addControl(ham, name="q0-ctrly", onQubits=0, matrices=Operator.driveY(level))
qham.addControl(ham, name="q0-ctrlz", onQubits=0, matrices=Operator.number(level))

# ------------------------------------------
# Run the optimization and show the results.
# ------------------------------------------

# Run the optimization.
ham, infidelity = opt(ham, FixedGate.H.getMatrix(), 40, xyzPulses=[1, 1, 0])

# Print infidelity and the waveform(s).
print(f"minimum infidelity: {infidelity}")
qham.plotWaves(ham, ["q0-ctrlx", "q0-ctrly", "q0-ctrlz"])

# Print the evolution process.
result = qham.simulate(ham)
projectedEvolution = project(result["unitary"], qubits, level, 2)
print("Projected evolution:\n", round(projectedEvolution, 2))

# Print the basic information of Hamiltonian.
qham.printHam(ham)
