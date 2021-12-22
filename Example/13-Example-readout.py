#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2020 Baidu, Inc. All Rights Reserved.
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
Example: readout simulation.
"""

from Quanlse.Superconduct.Simulator import ReadoutPulse
from Quanlse.Superconduct.Simulator.ReadoutSim3Q import readoutSim3Q
from math import pi
from matplotlib import pyplot as plt
from numpy import linspace


readoutModel = readoutSim3Q()

wr = readoutModel.resonatorFreq

driveFreq = {0: wr[0], 1: wr[1], 2: wr[2]}  # Drive frequency of the readout pulse.
driveStrength = {0: 0.001 * (2 * pi), 1: 0.001 * (2 * pi)}  # Drive strength of the readout pulse.
loFreq = 6.064 * (2 * pi)  # the local oscillator frequency for demodulation.

# Create a MeasChannel object and define the drive strength, drive frequency and local oscillator's frequency.
readoutPulse = ReadoutPulse(driveStrength=driveStrength, driveFreq=driveFreq, loFreq=loFreq)

# Define the measureChannel property.
readoutModel.readoutPulse = readoutPulse

# The range of the drive frequency to sweep.
freqsRange0 = wr[0] + 0.2 * linspace(-1, 1, 100) * (2 * pi)

freqsRange1 = wr[1] + 0.2 * linspace(-1, 1, 100) * (2 * pi)


q0Grd = []
i0Grd = []
q0Exd = []
i0Exd = []

q1Grd = []
i1Grd = []
q1Exd = []
i1Exd = []

rhoGrdList = []
rhoExdList = []

# Sweep the frequency when qubit is in ground state
for freq in freqsRange0:

    # Set the drive frequency of the readout pulse
    readoutModel.readoutPulse.setFreq(0, freq)
    readoutModel.readoutPulse.setFreq(1, freq)

    # Set the duration (in nanoseconds) and the indexes of the resonator simulated.
    data = readoutModel.simulate(duration=1000, resIdx=[0, 1], state='ground')

    vi = data['vi']
    vq = data['vq']

    rhoGrdList.append(data['rho'])

    i0Grd.append(vi[0])
    q0Grd.append(vq[0])

    i1Grd.append(vi[1])
    q1Grd.append(vq[1])

# Sweep the frequency when qubit is in excited state

for freq in freqsRange1:

    readoutModel.readoutPulse.setFreq(0, freq)
    readoutModel.readoutPulse.setFreq(1, freq)

    data = readoutModel.simulate(duration=1000, resIdx=[0, 1], state='excited')

    vi = data['vi']
    vq = data['vq']

    rhoExdList.append(data['rho'])

    i0Exd.append(vi[0])
    q0Exd.append(vq[0])

    i1Exd.append(vi[1])
    q1Exd.append(vq[1])


# Plot the IQ signal versus drive frequency.
plt.figure(1)

plt.subplot(221)

plt.plot(freqsRange0 * 1e3 / (2 * pi), i0Grd, label='I0, ground')
plt.plot(freqsRange0 * 1e3 / (2 * pi), i0Exd, label='I0, excited')
plt.xlabel('Drive frequency (MHz)')
plt.ylabel('Signal (a.u.)')

plt.legend()

plt.subplot(222)

plt.plot(freqsRange0 * 1e3 / (2 * pi), q0Grd, label='Q0, ground')
plt.plot(freqsRange0 * 1e3 / (2 * pi), q0Exd, label='Q0, excited')
plt.xlabel('Drive frequency (MHz)')
plt.ylabel('Signal (a.u.)')

plt.legend()

plt.subplot(223)

plt.plot(freqsRange1 * 1e3 / (2 * pi), i1Grd, label='I1, ground')
plt.plot(freqsRange1 * 1e3 / (2 * pi), i1Exd, label='I1, excited')
plt.xlabel('Drive frequency (MHz)')
plt.ylabel('Signal (a.u.)')

plt.legend()

plt.subplot(224)

plt.plot(freqsRange1 * 1e3 / (2 * pi), q1Grd, label='Q1, ground')
plt.plot(freqsRange1 * 1e3 / (2 * pi), q1Exd, label='Q1, excited')
plt.xlabel('drive frequency (MHz)')
plt.ylabel('Signal (a.u.)')

plt.legend()

plt.show()

x = linspace(-5, 5, 200)
y = linspace(-5, 5, 200)


