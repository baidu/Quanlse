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
Example: Zero-Noise Extrapolation
Zero-noise extrapolation (ZNE) is an increasingly popular technique for
mitigating errors in noisy quantum computations without using additional
quantum resources. In Quanlse, this technique is implemented.
You can access this service via the interface: remoteZNE.remoteZNEMitigation()
Please visit https://quanlse.baidu.com/#/doc/tutorial-ZNE for more details.
"""

import numpy as np
from copy import deepcopy
from Quanlse.ErrorMitigation.ZNE.Extrapolation import extrapolate
from Quanlse.ErrorMitigation.Utils.Visualization import plotZNESequences
from Quanlse.ErrorMitigation.Utils.Utils import computeIdealExpectationValue, \
    computeIdealEvolutionOperator, fromCircuitToHamiltonian, randomCircuit, \
    computeInverseGate
from Quanlse.Utils.Functions import project, expect
from Quanlse.Utils.Infidelity import unitaryInfidelity
from Quanlse import Define
from Quanlse.remoteSimulator import remoteSimulatorRunHamiltonian as runHamiltonian
from Quanlse.remoteZNE import remoteZNEMitigation as zneMitigation

# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# -------------------------------------------------
# Step 1. Formally describe the computational task.
# -------------------------------------------------

# Set the maximal length of the random Clifford circuit.
numSeq = 5
numQubits = 1

# Set the input state and the quantum observable both to |0><0|.
state = np.diag([1, 0]).astype(complex)
A = np.diag([1, 0]).astype(complex)

# Set the maximal extrapolation order.
order = 2

# Considering the reproducibility of our calculation result,
# we may as well set the "random seed" as a fixed value (e.g. 123).
circuit = randomCircuit(qubits=1, numSeq=numSeq, seed=123)

# Construct the identity-equivalent quantum circuit by appending an inverse gate to the end.
circuitIdentity = circuit + [computeInverseGate(circuit)]

# Compute the ideal expectation value (should be 1.0) and the ideal evolution operator.
valueIdeal = computeIdealExpectationValue(state, circuitIdentity, A)
unitaryIdeal = computeIdealEvolutionOperator(circuitIdentity)

# Compute the optimized Hamiltonian for implementing the quantum circuit.
# The built-in Quanlse Scheduler will be called.
ham = fromCircuitToHamiltonian(circuitIdentity)

# Use the optimized Hamiltonian to compute the implemented evolution unitary,
# the infidelity, and the noisy expectation value.
result = runHamiltonian(ham)
unitaryNoisy = project(result.result[0]["unitary"], ham.subSysNum, ham.sysLevel, 2)
infid = unitaryInfidelity(unitaryIdeal, unitaryNoisy, numQubits)
noisyValue = expect(A, unitaryNoisy @ state @ unitaryNoisy.conj().T)

# Print the ideal and noisy expectation values.
print("The ideal expectation value: {}; The noisy expectation: {}".format(valueIdeal, noisyValue))
print("The ideal evolutionary operator:")
print(unitaryIdeal.round(3))
print('The noisy evolutionary operator:')
print(unitaryNoisy.round(3))
print("The implemented evolution unitary has infidelity: ", infid)

# -----------------------------------------------------------------------
# Step 2. Use the ZNE method to improve the accuracy of expectation value.
# -----------------------------------------------------------------------

EsRescaled = []  # EsRescaled size: [numSeq, order + 1]
EsExtrapolated = []  # EsExtrapolated size: [numSeq, order]
EsIdeal = []  # EsIdeal size: [numSeq,]
Infidelities = []  # Infidelities size: [numSeq, order + 1]

for length in range(1, numSeq + 1):
    print('==' * 20)
    print("Clifford circuit length:", length)
    # For each sequence, append the equivalent-inverse gate of all the preceding quantum gates
    # For each sequence, its length becomes: [1, 2, ..., numSeq] + 1
    circuitPart = deepcopy(circuit[:length])
    lastGate = computeInverseGate(circuitPart)
    circuitPart.append(lastGate)

    # Compute ideal expectations firstly for subsequent comparison in figure.
    EsIdeal.append(computeIdealExpectationValue(state, circuitPart, A))

    # Temporary extrapolated values of each order for each-length circuit.
    mitigatedValues = []

    # Use the Scheduler to compute the optimal Hamiltonian for this circuit.
    ham = fromCircuitToHamiltonian(circuitPart)

    # Rescale order: [c_0, c_1, ..., c_d]; extrapolation order: d
    mitigatedValueHighest, infidelities, noisyValues = zneMitigation(state, circuitPart, A, ham=ham, order=order)

    # Rescale order: [c_0, c_1], [c_0, c_1, c_2], ...., [c_0, ..., c_{d-1}]
    # Loop: for d in [1, ..., d - 1]
    for d in range(1, order):
        mitigatedValue = extrapolate(infidelities[:(d + 1)], noisyValues[:(d + 1)], type='richardson', order=d)
        mitigatedValues.append(mitigatedValue)

    mitigatedValues.append(mitigatedValueHighest)

    EsExtrapolated.append(mitigatedValues)
    EsRescaled.append(noisyValues)
    Infidelities.append(infidelities)

# X-axis represents length of quantum circuit, Y-axis represents expectation values.
plotZNESequences(EsRescaled, EsExtrapolated, EsIdeal, fileName='zne-single-qubit-clifford')

# To better illustrate extrapolation technique, in the following we compute,
# the error mitigated values using only the 2-order and 3-order rescaled expectation values.
InfidelitiesPartial = np.array(Infidelities)[:, 1:]
EsRescaledPartial = np.array(EsRescaled)[:, 1:]
orderPartial = order - 1
EsExtrapolatedPartial = []  # size: [numSeq, orderPartial]
for i in range(numSeq):
    mitigatedValues = []
    for d in range(1, orderPartial + 1):
        mitigatedValue = extrapolate(InfidelitiesPartial[i][:(d + 1)], EsRescaledPartial[i][:(d + 1)],
                                     type='richardson', order=d)
        mitigatedValues.append(mitigatedValue)
    EsExtrapolatedPartial.append(mitigatedValues)

plotZNESequences(EsRescaledPartial, EsExtrapolatedPartial, EsIdeal, fileName='zne-single-qubit-clifford-2')
