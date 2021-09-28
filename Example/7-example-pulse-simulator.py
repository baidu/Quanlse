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
Example: Rabi oscillation by pulse-level simulator
Please visit https://quanlse.baidu.com/#/doc/tutorial-pulse-level-simulator for more details about this example.
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt

from Quanlse.QWaveform import QJob, QJobList
from Quanlse.QOperator import driveX
from Quanlse.QWaveform import gaussian
from Quanlse.Utils.Functions import basis, expect, dagger
from Quanlse.Simulator import PulseModel
from Quanlse import Define
from Quanlse.remoteSimulator import remoteSimulatorRunHamiltonian as runHamiltonian



# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# Duration time.
tg = 200

# Number of qubit(s).
qubits = 1

# System energy level.
level = 3

# Sampling period.
dt = 1.0

# --------------------------
# Define the qubit arguments.
# --------------------------

# Qubit frequency and anharmonicity.
qubitFreq = {0: 5.2 * (2 * pi)}
qubitAnharm = {0: -0.33 * (2 * pi)}

# Define the amplitude list.
dCoef = 0.04 * (2 * pi)
ampRabi = np.linspace(0, 0.7, 100)
amps = dCoef * ampRabi

# --------------------------------
# Construct the system Hamiltonian.
# --------------------------------

model = PulseModel(subSysNum=qubits,
                   sysLevel=level,
                   qubitFreq=qubitFreq,
                   qubitAnharm=qubitAnharm,
                   dt=dt)
ham = model.createQHamiltonian()

# Initialize the jobList.
jobList = QJobList(subSysNum=qubits, sysLevel=level, dt=dt, title='Rabi')

# Append job to jobList.
for amp in amps:
    wave = gaussian(tg, a=amp, tau=tg / 2, sigma=tg / 8)
    job = QJob(subSysNum=qubits, sysLevel=level, dt=dt)
    job.appendWave(driveX, 0, waves=wave)
    job = model.getSimJob(job)
    jobList.addJob(jobs=job)

# --------------------------------
# Run the simulation and show the results.
# --------------------------------

# Run the simulations.
result = runHamiltonian(ham=ham, state0=basis(level, 0), jobList=jobList)

# Define projector for calculating the population.
prj = basis(level, 0) @ dagger(basis(level, 0))

numList = []
coorList = []
for res in result.result:
    state = res['state']
    rho = state @ dagger(state)
    numList.append(1 - expect(prj, state))

# Plot the evolution of population.
plt.plot(ampRabi, numList, 'o', label='q')
plt.title('Rabi experiment')
plt.legend()
plt.show()
