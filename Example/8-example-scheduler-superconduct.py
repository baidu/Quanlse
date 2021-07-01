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
Example: Quanlse Scheduler
Please visit https://quanlse.baidu.com/#/doc/tutorial-scheduler for more details about this example.
"""

import numpy
from numpy import identity

from Quanlse.QOperation.FixedGate import X, H, CZ, CNOT
from Quanlse.Utils.Functions import tensor, basis, project
from Quanlse.Simulator.PulseSim3Q import pulseSim3Q
from Quanlse.Scheduler.Superconduct.DefaultPipeline import defaultPipeline
from Quanlse.remoteSimulator import remoteSimulatorRunHamiltonian as runHamiltonian
from Quanlse import Define


# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# ---------------------------------
# Construct the system Hamiltonian.
# ---------------------------------

# Sampling period.
dt = 0.5

# Instantiate the simulator object by a 3-qubit template.
model = pulseSim3Q(dt=dt)
sysLevel = model.sysLevel
qubitNum = model.subSysNum

# Set the scheduling strategy.
model.pipeline.addPipelineJob(defaultPipeline())

# Add pulse sequence.
H(model.Q[0])
H(model.Q[1])
CZ(model.Q[0], model.Q[1])
H(model.Q[1])
H(model.Q[2])
CZ(model.Q[1], model.Q[2])
H(model.Q[2])

# --------------------------------
# Get target unitary of the scheduler.
# --------------------------------

# Define layers and calculate the unitary of each layer.
Hm, CZm, Xm, IDm, CNOTm = H.getMatrix(), CZ.getMatrix(), X.getMatrix(), identity, CNOT.getMatrix()
layers = [
    tensor(Hm, IDm(4)),
    tensor(CNOTm, IDm(2)),
    tensor(IDm(2), CNOTm)
]

# Get the target unitary.
uGoal = IDm(2 ** qubitNum)
for mat in layers:
    uGoal = mat @ uGoal

# --------------------------------
# Run the simulation and show the results.
# --------------------------------

# Schedule the model.
job = model.schedule()
job.plot()

# Run the simulation.
uReal = runHamiltonian(ham=model.ham, job=job)[0]["unitary"]
stateReal = project(uReal, qubitNum, 3, 2) @ basis(2 ** qubitNum, 0)

# Calculate the difference.
stateGoal = uGoal @ basis(2 ** qubitNum, 0)
popReal = numpy.square(numpy.abs(stateReal))
popGoal = numpy.square(numpy.abs(stateGoal))
popDiff = numpy.abs(popReal - popGoal)
print("Difference:\n", popDiff)
print("popReal:\n", popReal)
print("popGoal:\n", popGoal)

# Print the maximum infidelity.
inf = numpy.max(popDiff)
print("Maximum inf:", inf)
