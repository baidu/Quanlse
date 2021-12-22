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

r"""
Example: Generate a robust MS gate pulse
A robust MS gate pulse is extremely important to large-scale quantum computing which can tolerate dephasing noise
and timing noise. In Quanlse, this technique is implemented.
Please visit https://quanlse.baidu.com/#/doc/tutorial-trapped-ion-robust-control for more details about this example.
"""

# Import Robust Mølmer-Sørensen pulse module
from Quanlse.TrappedIon.QIonSystem import QChain1D, QLaser
from Quanlse.TrappedIon.Optimizer.OptimizerIon import optimizeIonSymmetry
from Quanlse.TrappedIon.QIonTrajectory import noiseFeature, allAlphaComb
import numpy as np

# set experiment parameters
# ion trap
ionNumber = 10
indexIon = [0, 1]  # ion that interact with the laser pulse
mass = 171  # atom mass
omegaXY = 22.5e6  # unit: Hz
omegaZ = 3.8e6  # unit: Hz
temperature = 1e-6  # unit: K
# laser
waveLength = 369.75  # unit: nm
detuning = 3.804e6  # unit: Hz
laserAngle = np.pi / 2  # angle between two laser beams
tau = 2e-4  # unit: s
segments = 15  # laser pulse segments
omegaMax = 62.8e6  # unit: Hz

# generate the entity of ion chip and laser
ionChain = QChain1D(ionMass=mass,
                    ionNumber=ionNumber,
                    trapZ=omegaZ,
                    trapXY=omegaXY,
                    temperature=temperature)

ionLaser = QLaser(waveLength=waveLength,
                  laserAngle=laserAngle,
                  segments=segments, detuning=detuning,
                  maxRabi=omegaMax,
                  tg=tau)

# use the symmetry method to optimize the laser pulse sequence to be dephasing robust
dephasingNoise = 2e3  # unit: Hz
laserFinal = optimizeIonSymmetry(ionChip=ionChain,
                                 laser=ionLaser,
                                 indexIon=indexIon,
                                 noise=dephasingNoise)

# show noise features using plot function
timingNoise = 0.001
noiseFeature(ionChip=ionChain,
             laser=laserFinal,
             indexIon=indexIon,
             noise=dephasingNoise,
             timeNoise=timingNoise)

# show all alpha trajectory
allAlphaComb(ionChip=ionChain, laser=laserFinal, index=indexIon)
