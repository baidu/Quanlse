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
There are generally two ways in experiments for characterizing the performance of a quantum
computer in a superconducting platform: Quantum Process Tomography(QPT) and Randomized Benchmarking (RB)
. QPT can completely characterize a gate, decomposing a process into Pauli or Kraus operators,
but improving gates by QPT is complicated and resource-consuming. Also, State Preparation And Measurement
(SPAM) errors can be confused with process errors. However, RB is a concept of using randomization
methods for benchmarking quantum gates. It is a scalable and SPAM robust method for benchmarking the
full set of gates by a single parameter using randomization techniques. So it is useful to use RB for
a relatively simple and at least SPAM-robust benchmark, especially when the number of qubit increases.

For more details, please visit https://quanlse.baidu.com/#/doc/tutorial-randomized-benchmarking.
"""

import numpy as np
from numpy import ndarray
from functools import reduce

from Quanlse.QOperation.RotationGate import U
from Quanlse.QWaveform import QJob
from Quanlse.remoteSimulator import remoteSimulatorRunHamiltonian as runHamiltonian

from Quanlse.Utils.Functions import fromMatrixToAngles
from Quanlse.Utils.Clifford import clifford1q
from Quanlse.QPlatform import Error
from Quanlse.Superconduct.Simulator import PulseModel
from Quanlse.Superconduct.SchedulerSupport import SchedulerSuperconduct
from Quanlse.QOperation.FixedGate import FixedGateOP
from Quanlse.Utils.Infidelity import rhoInfidelity


def RB(model: PulseModel, targetQubitNum: int, initialState: ndarray, size: int, width: int,
       sche: SchedulerSuperconduct, dt: float, targetGate: FixedGateOP = None, interleaved: bool = False,
       isOpen: bool = False) -> float:
    r"""
    Return the sequence's average fidelity for different number of Clifford.

    :param model: the QHamiltonian object of the multi-qubit system.
    :param targetQubitNum: the index of the qubit being benchmarked.
    :param initialState: the initial state of the system.
    :param size: the number of Cliffords in a "width".
    :param width: the number of random sequence(s) for each "size".
    :param sche: the initiated scheduler.
    :param dt: AWG's sampling time.
    :param targetGate: the gate being benchmarked.
    :param interleaved: whether to use the interleaved benchmarking method.
    :param isOpen: whether to solve open system.
    :return: the average sequence fidelity of different number of Cliffords.
    """
    # Test inputs correctness
    if not isinstance(targetQubitNum, int):
        raise Error.ArgumentError("We only support single-qubit RB for now.")
    else:
        job = QJob(subSysNum=model.subSysNum, sysLevel=model.sysLevel, dt=dt)
        ham = model.createQHamiltonian()
        jobs = ham.createJobList()
        # Get the RB-target qubit
        ham.job = job
        hamTarget = ham.subSystem(targetQubitNum)
        sche.clearCircuit()
        finalFidelity = []
        # Generate random Clifford
        a = 0
        while a < width:
            idx = np.random.randint(0, 24, size)
            b = 0
            clifford1qInvReg = []
            # Schedule m Clifford gates
            while b < size:
                if not interleaved:
                    clifford1q[idx[size - b - 1]](sche.Q[0])
                    clifford1qInvReg.append(clifford1q[idx[b]].getMatrix())
                else:
                    targetGate(sche.Q[0])
                    clifford1q[idx[size - b - 1]](sche.Q[0])
                    clifford1qInvReg.append(clifford1q[idx[b]].getMatrix() @ targetGate.getMatrix())
                b += 1
            # Schedule inverse Clifford
            gateUnitary = np.transpose(np.conjugate(reduce(np.dot, clifford1qInvReg)))
            [_, theta, phi, lamda] = fromMatrixToAngles(gateUnitary)
            U(theta, phi, lamda)(sche.Q[0])
            # Schedule the whole RB sequence
            jobTarget = sche.schedule()
            hamTarget.job = jobTarget
            jobScheFinal = hamTarget.outputInverseJob(subSysNum=model.subSysNum, sysLevel=model.sysLevel, dt=dt)
            job = model.getSimJob(jobScheFinal)
            jobs.addJob(jobs=job)
            sche.clearCircuit()
            jobTarget.clearWaves()
            jobScheFinal.clearWaves()
            job.clearWaves()
            a += 1
        ham.buildCache()
        # Solve the ODE function
        if not isOpen:
            result = runHamiltonian(ham, state0=initialState, jobList=jobs)
        else:
            result = runHamiltonian(ham, state0=initialState, jobList=jobs, isOpen=True)
        rhoi = initialState @ ((initialState.transpose()).conjugate())
        for res in result.result:
            psit = res['state']
            rhot = psit @ ((psit.transpose()).conjugate())
            # Calculate the single sequence fidelity
            fidelity = 1 - np.real(rhoInfidelity(rhot, rhoi))
            finalFidelity.append(fidelity)
        ham.job.clearWaves()
        ham.clearCache()
        hamTarget.job.clearWaves()
        hamTarget.clearCache()
    return sum(finalFidelity) / width
