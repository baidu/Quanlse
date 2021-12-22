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
Generally, trapped ions in ion Chip are controlled by laser. The Hamiltonian
of large ion chain is a huge Hilbert space, and the quantum gate forms by
Schrodinger's equation is sometimes hard. But in Lamb-Dicke regime, we can get
the multi-qubit gates by Magnus expansion, and get the unitary form of general
trapped ion system:

:math:`U(\tau)=\sum_j\sum_{k=0}^{N-1}(\alpha_{j,k}(\tau)\hat{a}^\dagger+\alpha_{j,k}^*(\tau)\hat{a})\hat{\sigma}_\phi^{(j)}
+i\sum_{m\neq n}^N\chi_{m,n}(\tau)\hat{\sigma}_\phi^{(m)}\hat{\sigma}_\phi^{(n)}.`

And the evolution is described by phonon-ion coupling and effective ion-ion coupling.
below we define the basic element of how to get the evolution function in trapped ion system.
"""

from numpy import sin, cos, ndarray
import numpy as np
import math
from Quanlse.TrappedIon.QIonSystem import QChain1D, QLaser


def singleIntegral(args: list) -> tuple:
    r"""
    Analysis integral method of pulse_scheme
    :math:`\int_a^b \cos(\delta_k t - \phi) dt`
    :math:`\int_a^b \sin(\delta_k t - \phi) dt`

    :param args: input parameter, including the detuning, phase and integral range

    :return: integral value
    """

    differ, phi, a, b = args[0], args[1], args[2], args[3]
    analysisIntegralCos = sin(differ * b - phi) - sin(differ * a - phi)
    analysisIntegralSin = cos(differ * b - phi) - cos(differ * a - phi)
    return analysisIntegralCos / differ, -analysisIntegralSin / differ


def doubleIntegral1(args: list) -> float:
    r"""
    Analysis double integral method of pulse_scheme
    :math:`\int_a^b dt_1 \int_c^d dt_2 \sin(\delta_k (t_2-t_1) - \Delta\phi)`

    :param args: input parameter, including the detuning differ, phase phi and integral range

    :return: integral value
    """

    differ, phi = args[0], args[1]
    a, b, c, d = args[2], args[3], args[4], args[5]
    result = sin(-differ * b + differ * d - phi) - sin(-differ * a + differ * d - phi) - \
        sin(-differ * b + differ * c - phi) + sin(-differ * a + differ * c - phi)
    return result / (differ ** 2)


def doubleIntegral2(args: list) -> float:
    r"""
    Analysis double integral method of pulse_scheme
    :math:`\int_a^b dt_1 \int_c^{t_1} dt_2 \sin(\delta_k(t_2-t_1) - \Delta\phi)`

    :param args: input parameter, including the detuning differ, phase phi and integral range

    :return: integral value
    """

    differ, phi = args[0], args[1]
    a, b, c = args[2], args[3], args[4]
    result = - cos(-phi) * (b - a) / differ - sin(-differ * b + differ * c - phi) + sin(-differ * a + differ * c - phi)
    return result / (differ ** 2)


def alpha(seq: ndarray, detuning: float, tau: float) -> list:
    r"""
    Calculate phonon ion coupling for single ion and phonon
    :math:`\alpha_{j,m}(\tau)`

    :param seq: input pulse sequence like amplitude slice and phase slice
    :param detuning: the laser detuning
    :param tau: the pulse time

    :return: value of single :math:`\alpha_{j,m}`
    """
    omega, phi = seq[0], seq[1]
    result1, result2 = [], []
    tauSeg = tau / seq.shape[1]
    for index in range(seq.shape[1]):
        args = [detuning, phi[index], index * tauSeg, (index + 1) * tauSeg]
        integral1, integral2 = singleIntegral(args=args)
        result1.append(omega[index] / 2 * integral1)
        result2.append(omega[index] / 2 * integral2)
    return [sum(result1), sum(result2)]


def alphaIon(ionChip: QChain1D, laser: QLaser, indexM: int):
    """
    Function to give the phonon-ion coupling strength for a single ion

    :param ionChip: the ion chain system class
    :param laser: the controlled laser beam class
    :param indexM: the specific ion index we want to calculate
    :return: single sequence couping of phonon and ion
    """
    eta = np.array(ionChip.lambDickeAxial) * laser.waveVector

    frequency = ionChip.ionAxialFre*1.0
    detuning = laser.detuning
    tau = laser.tg
    seq = laser.seq[indexM]

    alphaI = np.zeros(eta.shape[1])
    alphaJ = np.zeros(eta.shape[1])
    for indexFre in range(eta.shape[1]):
        alphaI[[indexFre]], alphaJ[[indexFre]] = eta[[indexM], [indexFre]] * alpha(seq, frequency[indexFre] - detuning,
                                                                                   tau)
    return alphaI, alphaJ


def phononIonCoupling(ionChip: QChain1D, laser: QLaser, indexIon: list) -> float:
    r"""
    Calculate the phonon-ion coupling :math:`\alpha` strength

    :param ionChip: the ion chain system class
    :param laser: the controlled laser beam class
    :param indexIon: The index of ions that optically interacted by the laser
    :return: phonon-ion coupling strength
    """
    eta = np.array(ionChip.lambDickeAxial) * laser.waveVector
    alphaRealI, alphaImageI = alphaIon(ionChip, laser, indexIon[0])
    alphaRealJ, alphaImageJ = alphaIon(ionChip, laser, indexIon[1])
    alphaResult = np.zeros(eta.shape[0] * eta.shape[1]).reshape(eta.shape[0], eta.shape[1])
    for indexM in range(eta.shape[1]):
        alphaResult[0, [indexM]] = alphaRealI[indexM] ** 2 + alphaImageI[indexM] ** 2
        alphaResult[1, [indexM]] = alphaRealJ[indexM] ** 2 + alphaImageJ[indexM] ** 2
    return sum(sum(alphaResult))


def phononIonCouplingExact(ionChip: QChain1D, laser: QLaser, indexIon: list) -> float:
    r"""
    Calculate the exact phonon-ion coupling :math:`\alpha` with the regard to temperature

    :param ionChip: the ion chain system class
    :param laser: the controlled laser beam class
    :param indexIon: The index of ions that optically interacted by the laser
    :return: exact phonon-ion coupling strength :math:`\alpha`
    """
    eta = np.array(ionChip.lambDickeAxial) * laser.waveVector
    nk = ionChip.axialPhononPopulation
    beta = np.array([1 / math.tanh((1 / 2) * math.log(1 + 1 / nk[i])) for i in range(len(nk))])

    alphaRealI, alphaImageI = alphaIon(ionChip, laser, indexIon[0])
    alphaRealJ, alphaImageJ = alphaIon(ionChip, laser, indexIon[1])
    alphaResult = np.zeros(2 * eta.shape[1]).reshape(2, eta.shape[1])
    for index in range(eta.shape[1]):
        alphaResult[0, [index]] = (alphaRealI[index] ** 2 + alphaImageI[index] ** 2) * beta[index] / 2
        alphaResult[1, [index]] = (alphaRealJ[index] ** 2 + alphaImageJ[index] ** 2) * beta[index] / 2
    result = [np.exp(-sum(alphaResult[i])) for i in range(len(alphaResult))]
    return sum(result)


def twoIonCoupling(ionChip: QChain1D, laser: QLaser, indexIon: list) -> float:
    r"""
    Calculate the summation of all ion ion coupling
    :math:`\kappa_{j,j'}(\tau)`

    :param ionChip: the ion chain system class
    :param laser: the controlled laser beam class
    :param indexIon: The index of ions that optically interacted by the laser
    :return: value of single :math:`\kappa_{i,j}`
    """

    eta = np.array(ionChip.lambDickeAxial) * laser.waveVector
    seq = laser.seq
    tau = laser.tg
    frequency = ionChip.ionAxialFre
    detuning = laser.detuning

    omega1, omega2 = seq[indexIon[0]][0], seq[indexIon[1]][0]
    phi1, phi2 = seq[indexIon[0]][1], seq[indexIon[1]][1]
    result = 0.0
    tau = tau / omega1.shape[0]
    for indexM in range(frequency.shape[0]):
        for indexK in range(omega1.shape[0]):
            for indexL in range(indexK, omega2.shape[0]):
                if indexK < indexL:
                    args1 = [frequency[indexM] - detuning, phi2[indexL] - phi1[indexK], indexK * tau,
                             (indexK + 1) * tau, indexL * tau, (indexL + 1) * tau]
                    result = eta[indexIon[0]][indexM] * eta[indexIon[1]][indexM] * omega1[indexK] * omega2[
                        indexL] / 2 * doubleIntegral1(args=args1) + result
                else:
                    args2 = [frequency[indexM], phi2[indexL] - phi1[indexK], indexK * tau,
                             (indexK + 1) * tau, indexL * tau]
                    result = eta[indexIon[0]][indexM] * eta[indexIon[1]][indexM] * omega1[indexK] * omega2[
                        indexL] / 2 * doubleIntegral2(args=args2) + result
    return result


def getDiff(ionChip: QChain1D, laser: QLaser, indexIon: list) -> float:
    """
    Calculate the approximated difference term in the expression of infidelity

    :param ionChip: the ion chain system class
    :param laser: the controlled laser beam class
    :param indexIon: The index of ions that optically interacted by the laser

    :return: alpha differ term in approximated fidelity
    """

    eta = np.array(ionChip.lambDickeAxial) * laser.waveVector
    alphaRealI, alphaImageI = alphaIon(ionChip, laser, indexIon[0])
    alphaRealJ, alphaImageJ = alphaIon(ionChip, laser, indexIon[1])
    diffResults = np.zeros(eta.shape[1])
    for k in range(eta.shape[1]):
        diffResults[k] = (alphaRealI[k] - alphaRealJ[k]) ** 2 + (alphaRealI[k] + alphaRealJ[k]) ** 2 + (
                alphaImageI[k] - alphaImageJ[k]) ** 2 + (alphaImageI[k] - alphaImageJ[k]) ** 2
    return sum(diffResults)


def alphaExactDiff(ionChip: QChain1D, laser: QLaser, indexIon: list) -> float:
    """
    Calculate the exact difference term in the expression of infidelity with the regard to temperature

    :param ionChip: the ion chain system class
    :param laser: the controlled laser beam class
    :param indexIon: The index of ions that optically interacted by the laser
    :return: alpha differ term in exact fidelity
    """

    eta = np.array(ionChip.lambDickeAxial) * laser.waveVector
    nk = ionChip.axialPhononPopulation
    beta = np.array([1 / math.tanh((1 / 2) * math.log(1 + 1 / nk[i])) for i in range(len(nk))])

    alphaRealI, alphaImageI = alphaIon(ionChip, laser, indexIon[0])
    alphaRealJ, alphaImageJ = alphaIon(ionChip, laser, indexIon[1])
    diffResults = np.zeros(2 * eta.shape[1]).reshape(2, eta.shape[1])
    for k in range(eta.shape[1]):
        diffResults[0, k] = ((alphaRealI[k] - alphaRealJ[k]) ** 2 + (alphaImageI[k] - alphaImageJ[k]) ** 2) * beta[
            k] / 2
        diffResults[1, k] = ((alphaRealI[k] + alphaRealJ[k]) ** 2 + (alphaImageI[k] + alphaImageJ[k]) ** 2) * beta[
            k] / 2
    result = [np.exp(-sum(diffResults[i])) for i in range(len(diffResults))]
    return sum(result)
