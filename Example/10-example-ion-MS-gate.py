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
Example: Molmer-Sorensen gate in trapped ion
Please visit https://quanlse.baidu.com/#/doc/tutorial-ion-trap-single-and-two-qubit-gate
for more details about this example.
"""

from math import pi

from Quanlse.Utils import Plot
from Quanlse.remoteOptimizer import remoteIonMS as runIonMS
from Quanlse import Define


# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# ---------------------------------
# Construct the system Hamiltonian.
# ---------------------------------

# The parameters of the 1-d potential well.
omegaZ = 2 * pi * 0.2e6
omegaXY = 2 * pi * 4.1e6

# The number of ions of the 1-d chain.
ionNumber = 4
atomMass = 40

# Transverse or axial vibration phonon mode,
# phononMode = "axial".
phononMode = "transverse"

# The indexes of the qubits.
ionM = 1
ionN = 2

# Gate time of the Molmer-Sorensen gate.
tgate = 50

# The waveform of the laser，squareWave or sinWave
pulseWave = "squareWave"

# ------------------------------------------
# Run the simulation and show the results.
# ------------------------------------------

res, unitary = runIonMS(ionNumber=ionNumber,
                        atomMass=atomMass,
                        tg=tgate,
                        omega=(omegaXY, omegaZ),
                        ionIndex=(ionM, ionN),
                        pulseWave=pulseWave,
                        phononMode=phononMode)

print(f"The trapped ion axial phonon mode frequencies are:\n {res['phonon_freq']}\n")
print(f"The trapped ion axial Lamb-Dicke parameters are:\n {res['lamb_dicke']}\n")
print(f"infidelity is: {res['infidelity']}\n")
print(unitary)

ionPos = res['ion_pos']

Plot.plotIonPosition(ionPos)

Plot.plotPulse([res['time']], [res['omega']],
               title=['Sin pulse for Molmer-Sorensen gate in trapped ion'],
               xLabel=r'Time ($\mu$s)', yLabel=['Rabi frequency (a.u)'], color=['blue'])
