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

from Quanlse.QOperation.FixedGate import H, CZ
from Quanlse.Utils.Functions import basis
from Quanlse.Superconduct.Simulator.PulseSim2Q import pulseSim2Q
from Quanlse.Superconduct.SchedulerSupport.PipelineCenterAligned import centerAligned

from Quanlse import Define
from Quanlse.Utils.Functions import computationalBasisList
from Quanlse.Utils.Plot import plotBarGraph

# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# ---------------------------------
# Construct the system Hamiltonian.
# ---------------------------------

# Sampling period.
dt = 0.01

# Instantiate the simulator object by a 2-qubit template.
model = pulseSim2Q(dt=dt, frameMode='lab')
model.savePulse = False
sysLevel = model.sysLevel
qubitNum = model.subSysNum

# Set the center-aligned scheduling sctrategy
model.pipeline.addPipelineJob(centerAligned)

# Define circuit
H(model.Q[0])
H(model.Q[1])
CZ(model.Q[0], model.Q[1])
H(model.Q[0])


# --------------------------------
# Run the simulation and show the results.
# --------------------------------

# Schedule the model.
job = model.schedule()
job.plot()

# Calculate final state
finalState = model.simulate(
    job=job, state0=basis(model.sysLevel ** model.subSysNum, 0), shot=1000)

# Plot the population of computational basis
plotBarGraph(computationalBasisList(2, 3), finalState[0]["population"], 
             "Counts of the computational basis", "Computational Basis", "Counts")
