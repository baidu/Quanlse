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
In this page, we present the methods for accessing the Quanlse Cloud Service. At present,
we provide services for superconducting platform and ion trap platform.

For more details, please visit:

*Superconduct*

- Superconducting single-qubit gates: https://quanlse.baidu.com/#/doc/tutorial-single-qubit

- GRAPE (GRadient Ascent Pulse Engineering): https://quanlse.baidu.com/#/doc/tutorial-single-qubit

- Superconducting Cross-resonance gates: https://quanlse.baidu.com/#/doc/tutorial-cr

- Superconducting Controlled-Z gates: https://quanlse.baidu.com/#/doc/tutorial-cz

- Superconducting iSWAP gates: https://quanlse.baidu.com/#/doc/tutorial-iswap

*Ion Trap*

- Ion single/two-qubit gates: https://quanlse.baidu.com/#/doc/tutorial-ion-trap-single-and-two-qubit-gate

- Ion Mølmer-Sørensen gates: https://quanlse.baidu.com/#/doc/tutorial-general-MS-gate

"""

from Quanlse.QPlatform.Utilities import (
    numpyMatrixToDictMatrix,
    dictMatrixToNumpyMatrix
)
from Quanlse.QHamiltonian import QHamiltonian as QHam
from Quanlse.QWaveform import QJob

from Quanlse.QRpc import rpcCall

from typing import Dict, Any, List, Tuple, Union
from numpy import ndarray


def remoteOptimize1Qubit(ham: QHam, uGoal: Union[ndarray, List[ndarray]], targetInfid, depth=3) \
        -> Union[Tuple[QJob, float], Tuple[List[QJob], List[float]]]:
    """
    Optimize an arbitrary superconducting single-qubit gate by Quanlse cloud service.

    :param ham: the QHamiltonian object.
    :param uGoal: the goal unitary matrices for optimization.
    :param depth: maximum circuit depth (pulse number).
    :param targetInfid: target infidelity.
    :return: a tuple containing the QJob list and infidelity list.
    """
    args = [ham.dump(), numpyMatrixToDictMatrix(uGoal)]
    kwargs = {"targetInfid": targetInfid, "depth": depth}
    origin = rpcCall("1Qubit", args, kwargs)

    if isinstance(uGoal, list):
        return [QJob.load(o) for o in origin["job"]], origin["inf"]
    else:
        return QJob.load(origin["job"]), origin["inf"]


def remoteOptimize1QubitGRAPE(ham: QHam, uGoal: Union[ndarray, List[ndarray]], tg: int = 20, iterate: int = 150,
                              xyzPulses: List[int] = None) -> Union[Tuple[QJob, float], Tuple[List[QJob], List[float]]]:
    """
    Optimize a single-qubit gate using Gradient Ascent Pulse Engineering.

    :param ham: the QHamiltonian object.
    :param uGoal: the target unitary.
    :param tg: gate time.
    :param iterate: max number of iteration.
    :param xyzPulses: a list of three integers indicating the numbers of
                      Gaussian pulses superposed on x, y, z controls respectively.
    :return: a tuple containing the QJob list and infidelity list.
    """
    args = [ham.dump(), numpyMatrixToDictMatrix(uGoal)]
    kwargs = {
        "tg": tg,
        "iterate": iterate,
        "xyzPulses": xyzPulses
    }
    origin = rpcCall("1QubitGRAPE", args, kwargs)

    if isinstance(uGoal, list):
        return [QJob.load(o) for o in origin["job"]], origin["inf"]
    else:
        return QJob.load(origin["job"]), origin["inf"]


def remoteOptimizeCr(ham: QHam, aBound: Tuple[float, float] = None, tg: float = 200, maxIter: int = 5,
                     targetInfidelity: float = 0.01) -> Tuple[QJob, float]:
    """
    Optimize a superconducting Cross-Resonance gate by Quanlse Cloud Service.

    :param ham: the QHamiltonian object.
    :param aBound: the optimization bound of pulse amplitude.
    :param tg: gate time.
    :param maxIter: max number of iteration.
    :param targetInfidelity: the target infidelity.
    :return: a tuple containing the return Hamiltonian and infidelity.
    """
    args = [ham.dump()]
    kwargs = {
        "aBound": aBound,
        "tg": tg,
        "maxIter": maxIter,
        "targetInfidelity": targetInfidelity
    }
    origin = rpcCall("CR", args, kwargs)
    return QJob.load(origin["job"]), origin["inf"]


def remoteOptimizeCz(ham: QHam, aBound: Tuple[float, float] = None, tg: float = 200, maxIter: int = 5,
                     targetInfidelity: float = 0.01) -> Tuple[QJob, float]:
    """
    Optimize a superconducting Controlled-Z gate by Quanlse Cloud Service.

    :param ham: the QHamiltonian object.
    :param aBound: the optimization bound of pulse amplitude.
    :param tg: gate time.
    :param maxIter: max number of iteration.
    :param targetInfidelity: the target infidelity.
    :return: a tuple containing the return Hamiltonian and infidelity.
    """
    args = [ham.dump()]
    kwargs = {
        "aBound": aBound,
        "tg": tg,
        "maxIter": maxIter,
        "targetInfidelity": targetInfidelity
    }
    origin = rpcCall("CZ", args, kwargs)
    return QJob.load(origin["job"]), origin["inf"]


def remoteOptimizeISWAP(ham: QHam, aBound: Tuple[float, float] = None, tg: float = 200, maxIter: int = 5,
                        targetInfidelity: float = 0.01) -> Tuple[QJob, float]:
    """
    Optimize a superconducting iSWAP gate by Quanlse cloud service.

    :param ham: the QHamiltonian object.
    :param aBound: the optimization bound of pulse amplitude.
    :param tg: gate time.
    :param maxIter: max number of iteration.
    :param targetInfidelity: the target infidelity.
    :return: a tuple containing the return Hamiltonian and infidelity.
    """
    args = [ham.dump()]
    kwargs = {
        "aBound": aBound,
        "tg": tg,
        "maxIter": maxIter,
        "targetInfidelity": targetInfidelity
    }
    origin = rpcCall("ISWAP", args, kwargs)
    return QJob.load(origin["job"]), origin["inf"]


def remoteIonOptimize1Qubit(axial: str, theta: float, tg: float) -> Tuple[float, float, ndarray]:
    """
    Optimize a superconducting iSWAP gate by Quanlse cloud service.

    :param axial: the rotating axial, 'ionRx' or 'ionRy'.
    :param theta: the angle of the rotation operation.
    :param tg: gate time.
    :return: a tuple containing the return Hamiltonian and infidelity.
    """
    args = [axial, theta, tg]
    kwargs = {}
    origin = rpcCall("Ion1Qubit", args, kwargs)
    return origin["a"], origin["b"], dictMatrixToNumpyMatrix(origin["qam"], complex)


def remoteIonMS(ionNumber: int, atomMass: int, tg: float, omega: Tuple[float, float], ionIndex: Tuple[int, int],
                phononMode: str = 'axial', pulseWave: str = 'squareWave') -> Tuple[Any, Any]:
    """
    Generate the Molmer-Sorensen gate in trapped ion

    :param ionNumber: the number of ions.
    :param atomMass: the atomic mass of the ion.
    :param tg: gate time.
    :param omega: 1-dimensional angular frequency of the potential trap.
    :param ionIndex: the index of the two ions.
    :param phononMode: the mode of the phonon oscillation.
    :param pulseWave: the waveform of the laser.
    :return: dict type result and ndarray type unitary.
    """
    args = [ionNumber, atomMass, tg, omega, ionIndex]
    kwargs = {
        "phononMode": phononMode,
        "pulseWave": pulseWave,
    }
    origin = rpcCall("IonMS", args, kwargs)
    return origin['result'], dictMatrixToNumpyMatrix(origin["unitary"], complex)


def remoteIonGeneralMS(gatePair: List[List[int]], args1: Tuple[int, int, float, float, str],
                       args2: Tuple[int, float, float]) -> Tuple[Dict[str, Any], ndarray]:
    """
    Generate general Molmer-Sorensen gate and GHZ state in trapped ion

    :param gatePair: the gate pair in ion chain.
    :param args1:
     args1[0]: the number of ions;
     args1[1]: the atom mass or atom specie;
     args1[2]: the XY trapped potential frequency;
     args1[3]: the Z trapped potential frequency;
     args1[4]: the mode of the phonon oscillation.
    :param args2:
     args2[0]: the pulse sequence number;
     args2[1]: the sample time;
     args2[2]: the laser detuning.
    :return: dict type result and ndarray type unitary.
    """
    args = [gatePair, args1, args2]
    kwargs = {}
    origin = rpcCall("IonGeneralMS", args, kwargs)
    return origin['result'], dictMatrixToNumpyMatrix(origin["unitary"], complex)
