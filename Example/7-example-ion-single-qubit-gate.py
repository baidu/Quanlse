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
Example: Single qubit gate optimizer of trapped ion
Please visit https://quanlse.baidu.com/#/doc/tutorial-trapped-ion-gate for more details about this example.
"""

from math import pi


from Quanlse import Define
from Quanlse.Utils.Hamiltonian import plotWaves, getUnitary

from Quanlse.remoteOptimizer import remoteIonOptimize1Qubit as runIonOptimize1Qubit

# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# Rotating frame axial, use ionRx or ionRy.
axial = "ionRx"

# The angle of the rotation.
theta = pi / 2

# Gate time.
tgate = 2

# Run the Optimizer.
ham, infid = runIonOptimize1Qubit(axial, theta, tgate)

# Print the result and plot the pulse.
print(infid)
plotWaves(ham, xUnit=f'$\mu$s', yUnit='Rabi frequency \n (a.u)')
print(getUnitary(ham))

