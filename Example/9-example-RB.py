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
Example: Single-qubit Randomized Benchmarking
Please visit https://quanlse.baidu.com/#/doc/tutorial-randomized-benchmarking for more details about this example.
"""

from math import pi
from scipy.optimize import curve_fit

from Quanlse.Utils.Functions import basis, tensor
from Quanlse.QOperation import FixedGate
from Quanlse.Superconduct.Simulator import PulseModel
from Quanlse.Superconduct.SchedulerSupport import SchedulerSuperconduct
from Quanlse.Superconduct.SchedulerSupport.GeneratorRBPulse import SingleQubitCliffordPulseGenerator
from Quanlse.Utils.RandomizedBenchmarking import RB
from Quanlse import Define


# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# Number of qubit(s).
qubits = 2

# System energy level.
level = 3

# Sampling period.
dt = 1.0

# ---------------------------
# Define the qubit arguments.
# ---------------------------

qubitArgs = {
    "couplingMap": {(0, 1): 0.005 * (2 * pi)},  # Coupling of Q0 and Q1
    "wq0": 4.16 * (2 * pi),  # Frequency of Q0
    "wq1": 4.00 * (2 * pi),  # Frequency of Q1
    "anharm0": -0.22 * (2 * pi),  # Anharmonicity of Q0
    "anharm1": -0.21 * (2 * pi)  # Anharmonicity of Q1
}

# Define the input of PulseModel.
qubitFreq = {0: qubitArgs['wq0'], 1: qubitArgs['wq1']}
qubitAnharm = {0: qubitArgs['anharm0'], 1: qubitArgs['anharm1']}
t1List = {0: 5000}
t2List = {0: 2000}

# --------------------------------
# Construct the system Hamiltonian.
# --------------------------------

# Create a noisy virtual-QPU.
model = PulseModel(subSysNum=qubits,
                   sysLevel=level,
                   couplingMap=qubitArgs['couplingMap'],
                   qubitFreq=qubitFreq,
                   dt=dt,
                   qubitAnharm=qubitAnharm,
                   T1=t1List, T2=t2List, ampSigma=0.0001)

# Obtain the Hamiltonian of the pulse model.
ham = model.createQHamiltonian()

# The initial state of this simulator.
initialState = tensor(basis(3, 0), basis(3, 0))

# Get the target qubit's basic hardware information.
targetQubitNum = 0
hamTarget = ham.subSystem(targetQubitNum)

# The gate we want to benchmark.
targetGate = FixedGate.H

# Create a list to store the outcomes.
sizeSequenceFidelityBasic = []
sizeSequenceFidelityInterleaved = []

# Core parameters of an RB experiment.
size = [1, 10, 20, 50, 75, 100, 125, 150, 175, 200]
width = 5

# --------------------------------
# Start RB experiment.
# --------------------------------

# First get a basicRB curve used for reference.
print("*" * 50)
print(" Randonmized Benchmark")
print("*" * 50)

# Schedule those pulses.
sche = SchedulerSuperconduct(dt=dt, ham=hamTarget, generator=SingleQubitCliffordPulseGenerator(hamTarget))

# Get a basicRB curve used for reference.
for i in size:
    print("-" * 50)
    print("Size is", i)
    print("-" * 20, "Size is", i, "-" * 21)
    widthSequenceFidelityBasic = RB(model=model, targetQubitNum=targetQubitNum, initialState=initialState, size=i,
                                    width=width, sche=sche, dt=dt, interleaved=False, isOpen=False)
    sizeSequenceFidelityBasic.append(widthSequenceFidelityBasic)
    print(sizeSequenceFidelityBasic)

# Implement the interleavedRB to benchmark our Hadamard gate.
print("*" * 50)
print(" Interleaved Randonmized Benchmark")
print("*" * 50)
for j in size:
    print("-" * 50)
    print("Size is", j)
    print("-" * 20, "Size is", "-" * 21)

    widthSequenceFidelityInterleaved = RB(model=model, targetQubitNum=targetQubitNum, initialState=initialState,
                                          size=j, width=width, targetGate=targetGate, sche=sche, dt=dt,
                                          interleaved=True, isOpen=False)
    sizeSequenceFidelityInterleaved.append(widthSequenceFidelityInterleaved)
    print(sizeSequenceFidelityInterleaved)

# --------------------------------
# Fit the curve and calculate parameters.
# --------------------------------


# Define the fitting function.
def fit(x, a, p, b):
    """
    Define the fitting curve.
    """
    return a * (p ** x) + b


# Get the EPC(Error-rate Per Clifford) and p_{ref}.
fitparaBasic, fitcovBasic = curve_fit(fit, size, sizeSequenceFidelityBasic, p0=[0.5, 1, 0.5], maxfev=500000,
                                      bounds=[0, 1])
pfitBasic = fitparaBasic[1]
rClifford = (1 - pfitBasic) / 2
print('EPC = :', rClifford)

# Get the parameter p_{gate}.
fitparaInterleaved, fitcovInterleaved = curve_fit(fit, size, sizeSequenceFidelityInterleaved,
                                                  p0=[fitparaBasic[0], 1, fitparaBasic[2]],
                                                  maxfev=500000, bounds=[0, 1])
pfitInterleaved = fitparaInterleaved[1]
yfitBasic = fitparaBasic[0] * (pfitBasic ** size) + fitparaBasic[2]
yfitInterleaved = fitparaInterleaved[0] * (pfitInterleaved ** size) + fitparaInterleaved[2]


# Calculate the EPG(Error-rate Per Gate) with p_{gate} and p_{ref}.
def targetGateErrorRate(pGate, pRef, dimension):
    """
    Calculate the specific gate error rate.
    """
    return ((1 - (pGate / pRef)) * (dimension - 1)) / dimension


# Get the EPG(Error-rate Per Gate).
EPG = targetGateErrorRate(pfitInterleaved, pfitBasic, dimension=2)
print('EPG = : ', EPG)
