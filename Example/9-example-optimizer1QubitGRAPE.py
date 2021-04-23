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
Example: 1Qubit Gradient Ascent Pulse Engineering (GRAPE)
Please visit https://quanlse.baidu.com/#/doc/tutorial-GRAPE for more details about this example.
"""

import numpy as np
from math import pi
from Quanlse.Utils.Operator import driveX, driveY, number, duff
from Quanlse.Utils import Hamiltonian as qham
from Quanlse.QOperation import FixedGate
from Quanlse import Define

from Quanlse.remoteOptimizer import remoteOptimize1QubitGRAPE as runOptimize1QubitGRAPE


# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# --------------------------
# Define the input parameter.
# --------------------------
ugoal = FixedGate.X.getMatrix()
xyzPulses = [1, 0, 0]
iterate = 40


# --------------------------
# Define the input hamiltonian.
# --------------------------
dt = 1
level = 3
tg = 40
alphaq = - 0.22 * (2 * np.pi)  # anharmonicity
ham0 = qham.createHam(title="1q-3l", dt=dt, qubitNum=1, sysLevel=level)
qham.addDrift(ham0, "q0-anharm", onQubits=0, matrices=duff(level), amp=alphaq)
qham.addControl(ham0, name="q0-ctrlx", onQubits=0, matrices=driveX(level))
qham.addControl(ham0, name="q0-ctrly", onQubits=0, matrices=driveY(level))
qham.addControl(ham0, name="q0-ctrlz", onQubits=0, matrices=number(level))

# ------------------------------------------
# Run the optimization and show the results.
# ------------------------------------------
ham1, infid = runOptimize1QubitGRAPE(ham0, ugoal, iterate=iterate, tg=tg, xyzPulses=xyzPulses)

print(infid)
qham.plotWaves(ham1, ["q0-ctrlx", "q0-ctrly", "q0-ctrlz"], dark=True, color=['pink', 'lightblue', 'mint'])




