# !/usr/bin/python3
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
Example: Noisy simulation
Please visit https://quanlse.baidu.com/#/doc/tutorial-1qubit-simulator for more details about this example.
"""

from Quanlse.Utils.Waveforms import addPulse
from Quanlse.Utils.Plot import plotSched
import matplotlib.pyplot as plt
from Quanlse.remoteSimulator import remoteNoisySimulator as runNoisySimulator
from Quanlse import Define
import numpy as np


# Your token:
# Please visit http://quantum-hub.baidu.com
Define.hubToken = ''

# Parameters setting.
shots = 1024  # The execution time of the task.
anharm = -0.3472 * (2 * np.pi)  # The anharmonicity of the qubit.

deph_sigma = 0.01120487 #  The dephasing noise.
amp_gamma = 0.0159529  # The amplitude noise.

tg = 60  # The duration of the pulse.

# Define the pulses.
para1 = {'a': 0.6, 'tau': 0.5 * tg, 'sigma': tg / 8}  # The parameters of the pulse.
waveDataCtrl = []
waveDataCtrl.append(addPulse(channel='x', t0=0, t=tg, f='gaussian', para=para1))  # The control pulse.
waveDataCtrl.append(addPulse(channel='y', t0=tg, t=tg, f='gaussian', para=para1))
waveDataReadout = addPulse(channel='readout', t0=2 * tg, t=400, f='square', para={'a': 1.0})  # The readout pulse.
plotSched(waveDataCtrl=waveDataCtrl, waveDataReadout=waveDataReadout)  # Plot the pulse schedule defined.

# Run the simulator.
res = runNoisySimulator(waveDataCtrl=waveDataCtrl,
                        waveDataReadout=waveDataReadout,
                        dephSigma=deph_sigma,
                        ampGamma=amp_gamma,
                        anharm=anharm,
                        shots=shots)

# Plot the IQ datas.
plt.scatter(np.array(res['iq_data']['0'])[:, 0], np.array(res['iq_data']['0'])[:, 1], marker='.', label='0')
plt.scatter(np.array(res['iq_data']['1'])[:, 0], np.array(res['iq_data']['1'])[:, 1], marker='.', label='1')
plt.legend()
plt.show()

