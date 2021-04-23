#!/usr/bin/python3
# -*- coding: utf8 -*-
"""
remoteSimulator
"""

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
from Quanlse.Utils.Hamiltonian import toJson
from Quanlse.Utils.Waveforms import waveDataToSeq
from Quanlse.QPlatform.Utilities import dictMatrixToNumpyMatrix
from Quanlse.QRpc import rpcCall

from typing import Dict, Any, List, Union, Tuple, Callable


def remoteSimulatorRunHamiltonian(ham: Dict[str, Any], jobList: List[List[Dict[str, Any]]]) -> List[Dict]:
    """
    Given the Hamiltonian of a quantum system, this function returns the list of time-evolution unitary matrices
    using Quanlse Cloud Service.
    This function supports batch pulse simulation - users can submit a job list of multiple jobs.
    Each job is also a list, and multiple waves can be appended onto this list.
    On Quanlse Cloud Service, those waves will be added to ``ham``, and ``ham`` will be simulated;
    the result (including the time-evolution unitary matrix) will be append onto a result list corresponding
    to ``jobList``.

    :param ham: the Hamiltonian dictionary
    :param jobList: job list containing waveform lists
    :return: a list of result dictionary

    **Example: Scan the amplitudes**

    In this example, ``jobs`` is the list of jobs; ``jobWaves`` is the list of waves for each job. We use
    ``Quanlse.Utils.Waveforms.makeWaveData()`` to generate wave data, and append the wave data
    dictionary onto ``jobWaves``.

    .. code-block:: python

            jobs = []
            for amp in ampList:

                jobWaves = []

                # Add Gaussian wave of X control on the qubit 0
                jobWaves.append(makeWaveData(ham, "q0-ctrlx", t0=0, t=gateTime, f="gaussian", para={"a": amp}))
                jobWaves.append(makeWaveData(ham, "q1-ctrlx", t0=0, t=gateTime, f="gaussian", para={"a": amp}))

                # append this job onto the job list
                jobs.append(jobWaves)

    **Note:**

    Users need to import ``Quanlse.Define`` and set the token by the following code. Token can be acquired from the
    official website of Quantum Hub: http://quantum-hub.baidu.com.

    .. code-block:: python

            Define.hubToken = '...'

    """
    _jobListTransformed = []
    for job in jobList:
        _jobListTransformed.append(waveDataToSeq(job, ham["circuit"]["dt"]))
    args = [toJson(ham), _jobListTransformed]
    kwargs = {}
    origin = rpcCall("runHamiltonian", args, kwargs)
    for result in origin["resultBatch"]:
        result["unitary"] = dictMatrixToNumpyMatrix(result["unitary"], complex)
    return origin["resultBatch"]


def remoteNoisySimulator(waveDataCtrl: Union[Dict[str, Any], List[Dict[str, Any]]],
                         anharm: float = -2.1815, dephSigma: float = 0,
                         ampGamma: float = 0, kappa: float = 0.04,
                         waveDataReadout: Union[Dict[str, Any], List[Dict[str, Any]]]
                         = None, shots: int = 512) -> Dict[str, Any]:
    """
    Noisy simulator includes dephasing noise and amplitude noise
    with optional readout simulation.

    :param waveDataCtrl: A dictionary of waveData created by function 'addPulse()'.
    :param anharm: The anharmonicity of the qubit.
    :param dephSigma: Dephasing noise.
    :param ampGamma: Amplitude noise.
    :param kappa: The reasonator-environment interaction rate.
    :param waveDataReadout: A dictionary of waveData created by function 'addPulse()'.
    :param shots: The number of execution of the task.
    :return: A dictionary of the result, the keys are 'prob0', 'prob1', 'iq_data', 'counts'.
    """

    args = []
    kwargs = {
        "waveDataCtrl": waveDataToSeq(waveDataCtrl),
        "anharm": anharm,
        "dephSigma": dephSigma,
        "ampGamma": ampGamma,
        "kappa": kappa,
        "waveDataReadout": waveDataReadout,
        "shots": shots
    }

    origin = rpcCall("noisySimulator", args, kwargs)
    return origin["result"]
