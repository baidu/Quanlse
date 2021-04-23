#!/usr/bin/python3
# -*- coding: utf8 -*-
"""
remoteOptimizer
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

from Quanlse.Utils.Hamiltonian import toJson, createFromJson
from Quanlse.QPlatform.Utilities import numpyMatrixToDictMatrix, dictMatrixToNumpyMatrix
from Quanlse.QRpc import rpcCall

from typing import Dict, Any, List, Tuple
import numpy


def remoteOptimize1Qubit(ham: Dict[str, Any], uGoal: numpy.ndarray, tg: float = 20, xyzPulses: List[int] = None)\
        -> Tuple[Dict[str, Any], float]:
    """
    Optimize the drive pulses of an arbitrary single-qubit gate in superconducting quantum computing
    using Quanlse Cloud Service.


    **Note:**

    Users need to import ``Quanlse.Define`` and set the token by the following code. Token can be acquired from the
    official website of Quantum Hub: http://quantum-hub.baidu.com.

    .. code-block:: python

            Define.hubToken = '...'


    :param ham: the Hamiltonian dictionary
    :param uGoal: the target unitary matrices of the quantum gate
    :param tg: gate time
    :param xyzPulses: a list of three integers indicating the numbers of Gaussian pulses superposed on the x, y, z
        control respectively
    :return: a tuple containing the optimized Hamiltonian and gate infidelity
    """
    args = [toJson(ham), numpyMatrixToDictMatrix(uGoal)]
    kwargs = {"tg": tg, "xyzPulses": xyzPulses}
    origin = rpcCall("1Qubit", args, kwargs)
    return createFromJson(origin["qam"]), origin["infidelity"]


def remoteOptimizeCr(ham: Dict[str, Any], aBound: Tuple[float, float] = None, tg: float = 200, maxIter: int = 5,
                     targetInfidelity: float = 0.01) -> Tuple[Dict[str, Any], float]:
    """
    Optimize the drive pulses of the Cross-Resonance (CR) gate in superconducting quantum computing using
    Quanlse Cloud Service.


    **Note:**

    Users need to import ``Quanlse.Define`` and set the token by the following code. Token can be acquired from the
    official website of Quantum Hub: http://quantum-hub.baidu.com.

    .. code-block:: python

            Define.hubToken = '...'


    :param ham: the Hamiltonian dictionary
    :param aBound: the optimization bound of the pulse amplitude
    :param tg: gate time
    :param maxIter: maximum number of iteration
    :param targetInfidelity: the target gate infidelity
    :return: a tuple containing the optimized Hamiltonian and the gate infidelity
    """
    args = [toJson(ham)]
    kwargs = {
        "aBound": aBound,
        "tg": tg,
        "maxIter": maxIter,
        "targetInfidelity": targetInfidelity
    }
    origin = rpcCall("CR", args, kwargs)
    return createFromJson(origin["qam"]), origin["infidelity"]


def remoteOptimizeCz(ham: Dict[str, Any], aBound: Tuple[float, float] = None, tg: float = 40, maxIter: int = 5,
                     targetInfidelity: float = 0.01) -> Tuple[Dict[str, Any], float]:
    """
    Optimize the driven pulses of the controlled-Z (CZ) gate in superconducting quantum computing using
    Quanlse Cloud Service.


    **Note:**

    Users need to import ``Quanlse.Define`` and set the token by the following code. Token can be acquired from the
    official website of Quantum Hub: http://quantum-hub.baidu.com.

    .. code-block:: python

            Define.hubToken = '...'

    :param ham: the Hamiltonian dictionary
    :param aBound: the optimization bound of the pulse amplitude
    :param tg: gate time
    :param maxIter: maximum number of iteration
    :param targetInfidelity: the target gate infidelity
    :return: a tuple containing the optimized Hamiltonian and the gate infidelity
    """
    args = [toJson(ham)]
    kwargs = {
        "aBound": aBound,
        "tg": tg,
        "maxIter": maxIter,
        "targetInfidelity": targetInfidelity
    }
    origin = rpcCall("CZ", args, kwargs)
    return createFromJson(origin["qam"]), origin["infidelity"]


def remoteOptimizeISWAP(ham: Dict[str, Any], aBound: Tuple[float, float] = None, tg: float = 40, maxIter: int = 5,
                        targetInfidelity: float = 0.01) -> Tuple[Dict[str, Any], float]:
    """
    Optimize the drive pulses of the ISWAP gate in superconducting quantum computing using Quanlse Cloud Service.

    **Note:**

    Users need to import ``Quanlse.Define`` and set the token by the following code. Token can be acquired from the
    official website of Quantum Hub: http://quantum-hub.baidu.com.

    .. code-block:: python

            Define.hubToken = '...'

    :param ham: the Hamiltonian dictionary
    :param aBound: the optimization bound of pulse amplitude
    :param tg: gate time
    :param maxIter: maximum number of iteration
    :param targetInfidelity: the target gate infidelity
    :return: a tuple containing the optimized Hamiltonian and gate infidelity
    """
    args = [toJson(ham)]
    kwargs = {
        "aBound": aBound,
        "tg": tg,
        "maxIter": maxIter,
        "targetInfidelity": targetInfidelity
    }
    origin = rpcCall("ISWAP", args, kwargs)
    return createFromJson(origin["qam"]), origin["infidelity"]


def remoteOptimize1QubitGRAPE(ham: Dict[str, Any], uGoal: numpy.ndarray,
                              tg: int = 20, iterate: int = 150,
                              xyzPulses: List[int] = None)\
                              -> Tuple[Dict[str, Any], float]:
    """
    Optimize a 1-qubit gate using Gradient Ascent Pulse Engineering.

    **Note:**

    Users need to import ``Quanlse.Define`` and set the token by the following code. Token can be acquired from the
    official website of Quantum Hub: http://quantum-hub.baidu.com.

    .. code-block:: python

            Define.hubToken = '...'

    :param ham: the Hamiltonian dictionary.
    :param uGoal: the target unitary.
    :param tg: gate time.
    :param iterate: max number of iteration.
    :param xyzPulses: a list of three integers indicating the numbers of
                Gaussian pulses superposed on x, y, z controls respectively.
    :return: a tuple containing the return Hamiltonian and infidelity.
    """
    args = [toJson(ham), numpyMatrixToDictMatrix(uGoal)]
    kwargs = {
        "tg": tg,
        "iterate": iterate,
        "xyzPulses": xyzPulses
    }
    origin = rpcCall("1QubitGRAPE", args, kwargs)
    return createFromJson(origin["qam"]), origin["infidelity"]


def remoteIonOptimize1Qubit(axial: str, theta: float, tgate: float)\
                            -> Tuple[Dict[str, Any], float]:
    """
    Optimize a superconducting iSWAP gate by Quanlse cloud service.

    **Note:**

    Users need to import ``Quanlse.Define`` and set the token by the following code. Token can be acquired from the
    official website of Quantum Hub: http://quantum-hub.baidu.com.

    .. code-block:: python

            Define.hubToken = '...'

    :param axial: the rotating axial, 'ionRx' or 'ionRy'.
    :param theta: the angle of the rotation operation.
    :param tgate: gate time.
    :return: the unitary of the evolution.
    """
    args = [axial, theta, tgate]
    kwargs = {}
    origin = rpcCall("Ion1Qubit", args, kwargs)
    return createFromJson(origin["qam"]), origin["infidelity"]


def remoteIonMS(ionNumber: int, atomMass: int, tgate: float,
                omega: Tuple[float, float], ionIndex: Tuple[int, int],
                phononMode: str = 'axial', pulseWave: str = 'squareWave')\
                -> Tuple[Dict[str, Any], numpy.ndarray]:
    """
    Generate the Molmer-Sorensen gate in trapped ions by Quanlse cloud service.

    **Note:**

    Users need to import ``Quanlse.Define`` and set the token by the following code. Token can be acquired from the
    official website of Quantum Hub: http://quantum-hub.baidu.com.

    .. code-block:: python

            Define.hubToken = '...'

    :param ionNumber: the number of ions.
    :param atomMass: the atomic mass of the ion.
    :param tgate: gate time.
    :param omega: 1-dimensional angular frequency of the potential trap.
    :param ionIndex: the index of the two ions.
    :param phononMode: the mode of the phonon oscillation.
    :param pulseWave: the waveform of the laser.
    :return: the time sequency of the pulse.
    """
    args = [ionNumber, atomMass, tgate, omega, ionIndex]
    kwargs = {
        "phononMode": phononMode,
        "pulseWave": pulseWave,
    }
    origin = rpcCall("IonMS", args, kwargs)
    return origin['result'], dictMatrixToNumpyMatrix(origin["unitary"], complex)
