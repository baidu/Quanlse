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
Simulate the Hamiltonian using the Quanlse remote simulator.
"""

from numpy import ndarray

from Quanlse.QHamiltonian import QHamiltonian as QHam
from Quanlse.QWaveform import QJobList, QResult, QJob
from Quanlse.QRpc import rpcCall
from Quanlse.QPlatform.Utilities import numpyMatrixToDictMatrix


def remoteSimulatorRunHamiltonian(ham: QHam, state0: ndarray = None, job: QJob = None,
                                  jobList: QJobList = None, isOpen=False) -> QResult:
    """
    Simulate the Hamiltonian using the Quanlse remote simulator.

    :param ham: the QHamiltonian object.
    :param state0: The initial state.
    :param job: the QJob object.
    :param jobList: The QJobList object.
    :param isOpen: Run the simulation of open system using Lindblad master equation if true.
    :return: the QResult object.
    """

    maxEndTime = None
    kwargs = {}
    if state0 is not None:
        kwargs["state0"] = numpyMatrixToDictMatrix(state0)
    if job is not None:
        maxEndTime, _ = job.computeMaxTime()
        kwargs["job"] = job.dump(maxEndTime=maxEndTime)
    if jobList is not None:
        maxTimeList = []
        for job in jobList.jobs:
            maxTime, _ = job.computeMaxTime()
            maxTimeList.append(maxTime)
        maxEndTime = max(maxTimeList)
        kwargs["jobList"] = jobList.dump(maxEndTime=maxEndTime)
    if maxEndTime is None:
        maxEndTime, _ = ham.job.computeMaxTime()
    kwargs["isOpen"] = isOpen

    args = [ham.dump(maxEndTime=maxEndTime)]
    origin = rpcCall("runHamiltonian", args, kwargs)

    return QResult.load(origin["result"])
