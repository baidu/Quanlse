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
We offer the plot functions for many features in MS gate, including laser pulse scheme containing :math:`\Omega`
and :math:`\phi` sequences, phonon-ion coupling strength :math:`\alpha` and two ion coupling strength
:math:`\kappa` varied with time. Users can draw single or tuple trajectories using different functions.
"""

from Quanlse.TrappedIon.Optimizer import QIonEvolution, OptimizerIon
from Quanlse.TrappedIon.QIonSystem import QChain1D, QLaser
import numpy as np
from numpy import cos, sin, ndarray
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import integrate
import math


def funcRealAlpha(x, a, b, o):
    """
    Integrate function represented the real part of phonon-ion coupling strength

    :param x: time
    :param a: detuning
    :param b: phase
    :param o: phonon frequency
    :return: function for alpha real part integration
    """

    return math.sin(a * x - b) * math.cos(o * x)


def funcImgAlpha(x, a, b, o):
    """
    Integrate function represented the imaginary part of phonon-ion coupling strength

    :param x: time
    :param a: detuning
    :param b: phase
    :param o: phonon frequency
    :return: function for alpha imaginary part integration
    """

    return math.sin(a * x - b) * math.sin(o * x)


def alphaPlot(ionChip: QChain1D, laser: QLaser, index: list, n: int = 100):
    """
    Plot the variation trajectory of a specific phonon-ion coupling strength

    :param ionChip: ion chip class
    :param laser: laser class
    :param index: interact ion index
    :param n: segments
    """
    mu = laser.detuning
    omega = laser.seq[index[0]][0]
    phase = laser.seq[index[0]][1]
    tau = laser.tg
    axialFre = ionChip.ionAxialFre[index[0]]
    eta = ionChip.lambDickeAxial[index[0]][index[1]] * laser.waveVector

    pieces = len(omega)
    sec = tau / (n * pieces)
    alphaReal = np.array([[0.0, 0.0] for i in range(n * pieces + 1)])  # real part
    alphaImg = np.array([[0.0, 0.0] for i in range(n * pieces + 1)])  # imaginary part
    phase = phase.repeat(n)
    omega = omega.repeat(n)

    for i in range(n * pieces):
        index = i + 1

        alphaReal[index] = alphaReal[i] + omega[i] * np.array(
            integrate.quad(lambda x: funcRealAlpha(x, mu, phase[i], axialFre), i * sec, index * sec))  # real part
        alphaImg[index] = alphaImg[i] + omega[i] * np.array(
            integrate.quad(lambda x: funcImgAlpha(x, mu, phase[i], axialFre), i * sec, index * sec))  # imaginary part

    alphaRealSeq = eta * alphaReal.T[0]
    alphaImgSeq = eta * alphaImg.T[0]
    print([alphaRealSeq, alphaImgSeq])
    plt.plot(alphaRealSeq, alphaImgSeq)
    plt.title("trajectory for alpha")
    plt.xlabel("real part")
    plt.ylabel("imaginary part")
    plt.show()
    print('finish')


def alphaValue(mu: float, phase: ndarray, axialFre: float, a: float, b: float) -> [float, float]:
    """
    Function to calculate phonon-ion coupling strength in a specific time

    :param mu: input detuning
    :param phase: input pulse phase sequence
    :param axialFre: phonon frequency
    :param a: lower bound
    :param b: upper bound
    :return: analyzed integral value
    """
    real = math.sin((axialFre - mu) * b - phase) - math.sin((axialFre - mu) * a - phase)
    img = math.cos((axialFre - mu) * b - phase) - math.cos((axialFre - mu) * a - phase)
    return real / (axialFre - mu), -img / (axialFre - mu)


def alphaValueList(mu: float, omega: ndarray, phase: ndarray, tau: float, omega_k: float, eta: ndarray, n: int) -> list:
    """
    Function to give a phonon-ion coupling strength list over time

    :param n: sampling number in every segments
    :param mu: input detuning
    :param omega: input pulse amplitude sequence
    :param phase: input pulse phase sequence
    :param tau: gate duration
    :param omega_k: phonon frequency
    :param eta: Lamb-Dicke parameters
    :return: draw trajectory of alpha
    """

    # parameters for calculating
    pieces = len(omega)
    sec = tau / (n * pieces)
    alphaReal = np.array([[0.0, 0.0] for i in range(n * pieces + 1)])  # real part
    alphaImg = np.array([[0.0, 0.0] for i in range(n * pieces + 1)])  # imaginary part
    phase = phase.repeat(n)
    omega = omega.repeat(n)

    for i in range(n * pieces):
        index = i + 1
        real, img = alphaValue(mu, phase[i], omega_k, i * sec, index * sec)
        alphaReal[index] = alphaReal[i] + omega[i] * real  # real part
        alphaImg[index] = alphaImg[i] + omega[i] * img  # imaginary part

    alphaRealSeq = eta * alphaReal.T[0] / 2
    alphaImgSeq = eta * alphaImg.T[0] / 2
    return [alphaRealSeq, alphaImgSeq]


def allAlphaSep(ionChip: QChain1D, laser: QLaser, index: list):
    """
    Plot function of every phonon-ion coupling strength that distributed separately

    :param index: optically interacted ions
    :param ionChip: ion chip class
    :param laser: laser class
    """

    # extract parameters for calculating
    eta = np.array([ionChip.lambDickeAxial[i] * laser.waveVector for i in index])
    axialFre = ionChip.ionAxialFre
    sequence = np.array([laser.seq[i] for i in index])
    detuning = laser.detuning
    tau = laser.tg

    # plot alpha separately
    allAlphaSeq = [[[] for i in range(len(axialFre))] for j in range(len(sequence))]
    fig, axs = plt.subplots(len(sequence), len(axialFre), figsize=(3 * len(axialFre), 3 * len(sequence)))
    for ionIndex in range(len(sequence)):
        for phononIndex in range(len(axialFre)):
            alphaSingle = alphaValueList(detuning, sequence[ionIndex][0], sequence[ionIndex][1], tau,
                                         axialFre[phononIndex], eta[ionIndex][phononIndex], 200)
            allAlphaSeq[ionIndex][phononIndex].append(alphaSingle[0])
            allAlphaSeq[ionIndex][phononIndex].append(alphaSingle[1])
            axs[ionIndex, phononIndex].plot(allAlphaSeq[ionIndex][phononIndex][0],
                                            allAlphaSeq[ionIndex][phononIndex][1])
            axs[ionIndex, phononIndex].set_title(r"$\alpha_{%d%d}$" % (ionIndex, phononIndex))
    plt.show()


def allAlphaComb(ionChip: QChain1D, laser: QLaser, index: list):
    """
    Plot function of every phonon-ion coupling strength that plotted in one canvas

    :param index: optically interacted ions
    :param ionChip: ion chip class
    :param laser: laser class
    :return: all alpha trajectories plotted
    """

    # extract parameters for calculating
    eta = np.array([ionChip.lambDickeAxial[i] * laser.waveVector for i in index])
    axialFre = ionChip.ionAxialFre
    sequence = np.array([laser.seq[i] for i in index])
    detuning = laser.detuning
    tau = laser.tg

    # plot alpha together
    allAlphaSeq = [[[] for i in range(len(axialFre))] for j in range(len(sequence))]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for ionIndex in range(len(sequence)):
        axs[ionIndex].set_title(r"$\alpha_{%d}$" % ionIndex)
        for phononIndex in range(len(axialFre)):
            alphaSingle = alphaValueList(detuning, sequence[ionIndex][0], sequence[ionIndex][1], tau,
                                         axialFre[phononIndex], eta[ionIndex][phononIndex], 200)
            allAlphaSeq[ionIndex][phononIndex].append(alphaSingle[0])
            allAlphaSeq[ionIndex][phononIndex].append(alphaSingle[1])
            axs[ionIndex].plot(allAlphaSeq[ionIndex][phononIndex][0], allAlphaSeq[ionIndex][phononIndex][1])
            axs[ionIndex].set_xlabel("Real part")
            axs[ionIndex].set_ylabel("Imag part")
    plt.show()


def doubleIntegral1(args: list) -> float:
    r"""
    Analysis double integral method of pulse_scheme
    :math:`\int_a^b dt_1 \int_c^d dt_2 \sin(\delta_k(t_2-t_1) - \Delta\phi)`

    :param args: input parameter, including the detuning, phase and integral range

    :return: integral value
    """

    # extract parameters for calculating
    detuning, phi = args[0], args[1]
    a, b, c, d = args[2], args[3], args[4], args[5]
    twoIonCoupleAnalyze = sin(-detuning * b + detuning * d - phi) - sin(-detuning * a + detuning * d - phi) - sin(
        -detuning * b + detuning * c - phi) + sin(-detuning * a + detuning * c - phi)
    return twoIonCoupleAnalyze / (detuning ** 2)


def doubleIntegral2(args: list) -> float:
    r"""
    Analysis double integral method of pulse_scheme
    :math:`\int_a^b dt_1 \int_c^{t_1} dt_2 \sin(\delta_k(t_2-t_1) - \Delta\phi)`

    :param args: input parameter, including the detuning, phase and integral range

    :return: integral value
    """

    # extract parameters for calculating
    detuning, phi = args[0], args[1]
    a, b, c = args[2], args[3], args[4]
    twoIonCoupleAnalyze = - cos(-phi) * (b - a) / detuning - sin(-detuning * b + detuning * c - phi) + sin(
        -detuning * a + detuning * c - phi)
    return twoIonCoupleAnalyze / (detuning ** 2)


def twoIonCouplingPlot(ionChip: QChain1D, laser: QLaser, index: list, drawPieces: int = 10) -> float:
    r"""
    Calculate the summation of all ion ion coupling

    :param index: optically interacted ions
    :param ionChip: ion chip class
    :param laser: laser class
    :param drawPieces: sampling number in every segments

    :return: value of single `\alpha_{j,m}`
    """

    # extract parameters for calculating
    eta = np.array([ionChip.lambDickeAxial[i] for i in index]) * laser.waveVector
    axialFre = ionChip.ionAxialFre
    laserSeq = np.array([laser.seq[i] for i in index])
    detuning = laser.detuning
    tau = laser.tg
    omegaSeq1 = np.array(laserSeq[0][0]).repeat(drawPieces)
    omegaSeq2 = np.array(laserSeq[1][0]).repeat(drawPieces)
    phaseSeq1 = np.array(laserSeq[0][1]).repeat(drawPieces)
    phaseSeq2 = np.array(laserSeq[1][1]).repeat(drawPieces)
    tau = tau / omegaSeq1.shape[0]
    kappa = [[] for i in range(len(axialFre))]
    print([tau, detuning, laserSeq, axialFre, eta])

    # get the integration of two ion coupling in every time segments
    for indexM in range(axialFre.shape[0]):
        for indexK in range(omegaSeq1.shape[0]):
            result = 0.0
            for indexL in range(indexK, omegaSeq2.shape[0]):
                if indexK < indexL:
                    args1 = [axialFre[indexM] - detuning, phaseSeq2[indexL] - phaseSeq1[indexK], indexK * tau,
                             (indexK + 1) * tau, indexL * tau, (indexL + 1) * tau]
                    result = eta[0][indexM] * eta[1][indexM] * omegaSeq1[indexK] * omegaSeq2[
                        indexL] / 2 * doubleIntegral1(args=args1) + result
                else:
                    args2 = [axialFre[indexM], phaseSeq2[indexL] - phaseSeq1[indexK], indexK * tau,
                             (indexK + 1) * tau, indexL * tau]
                    result = eta[0][indexM] * eta[1][indexM] * omegaSeq1[indexK] * omegaSeq2[
                        indexL] / 2 * doubleIntegral2(args=args2) + result
            kappa[indexM].append(result)
    y = sum(np.array(kappa))
    y = [sum(y[:i + 1]) for i in range(len(y))]
    plt.title("Ion-ion coupling dynamics")
    plt.xlabel(r"Time ($\mu$s)")
    plt.ylabel(r"Ion-ion coupling strength $\chi$")
    plt.plot(range(len(y)), y)
    plt.show()
    return y[len(y) - 1]


def twoIonCouplingSeq(eta: ndarray, axialFre: ndarray, laserSeq: ndarray, detuning: float, tau: float,
                      drawPieces: int) -> list:
    r"""
    Calculate the summation of all ion ion coupling
    :math:`\kappa_{j,j'}(t_g)`

    :param eta: the input lamb-dicke parameter
    :param axialFre: the experiment phonon frequency
    :param laserSeq: the input pulse for ion j1 and j2
    :param detuning: the laser detuning
    :param tau: the pulse slice time
    :param drawPieces: sampling number in every segments

    :return: value of single :math:`\alpha_{j,m}`
    """

    # extract parameters for calculating
    omegaSeq1 = np.array(laserSeq[0][0]).repeat(drawPieces)
    omegaSeq2 = np.array(laserSeq[1][0]).repeat(drawPieces)
    phaseSeq1 = np.array(laserSeq[0][1]).repeat(drawPieces)
    phaseSeq2 = np.array(laserSeq[1][1]).repeat(drawPieces)
    tau = tau / omegaSeq1.shape[0]
    kappa = [[] for i in range(len(axialFre))]
    for indexM in range(axialFre.shape[0]):
        for indexK in range(omegaSeq1.shape[0]):
            result = 0.0
            for indexL in range(indexK, omegaSeq2.shape[0]):
                if indexK < indexL:
                    args1 = [axialFre[indexM] - detuning, phaseSeq2[indexL] - phaseSeq1[indexK], indexK * tau,
                             (indexK + 1) * tau, indexL * tau, (indexL + 1) * tau]
                    result = eta[0][indexM] * eta[1][indexM] * omegaSeq1[indexK] * omegaSeq2[
                        indexL] / 2 * doubleIntegral1(args=args1) + result
                else:
                    args2 = [axialFre[indexM], phaseSeq2[indexL] - phaseSeq1[indexK], indexK * tau,
                             (indexK + 1) * tau, indexL * tau]
                    result = eta[0][indexM] * eta[1][indexM] * omegaSeq1[indexK] * omegaSeq2[
                        indexL] / 2 * doubleIntegral2(args=args2) + result
            kappa[indexM].append(result)
    y = sum(np.array(kappa))
    y = [sum(y[:i + 1]) for i in range(len(y))]
    return y


def dephasingNoise(ionChip: QChain1D, laser: QLaser, index: list, noise: float) -> list:
    """
    Function that calculate the exact infidelity and two ion coupling strength under the dephasing noise.

    :param index: optically interacted ions
    :param ionChip: ion chip class
    :param laser: laser class
    :param noise: specific noise
    :return: plot variable of infidelity and kappa along noise
    """

    # test the robust characteristic curve
    detuningNoiseSeq = np.arange(-15, 16) * noise / 15
    detuning = laser.detuning
    infidelitySeq = []
    kappaSeq = []
    for noiseIndex in range(detuningNoiseSeq.shape[0]):
        laser.detuning = detuning + detuningNoiseSeq[noiseIndex]
        infidelityNoise = OptimizerIon.exactInfidelity(ionChip, laser, index)
        kappa = QIonEvolution.twoIonCoupling(ionChip, laser, index)
        infidelitySeq.append(infidelityNoise)
        kappaSeq.append(kappa)
        laser.detuning = detuning
    return [[detuningNoiseSeq * 1e-3, infidelitySeq], [detuningNoiseSeq * 1e-3, kappaSeq]]


def timingNoise(ionChip: QChain1D, laser: QLaser, index: list, timeNoise: float) -> list:
    """
    Function that calculate the exact infidelity and two ion coupling strength under the timing noise.

    :param index: optically interacted ions
    :param ionChip: ion chip class
    :param laser: laser class
    :param timeNoise: specific timing noise
    :return: plot variable of infidelity and kappa along noise
    """

    # test the robust characteristic curve
    timingNoiseSeq = np.arange(-30, 31) * timeNoise / 30 + np.ones(61)
    tau = laser.tg
    infidelitySeq = []
    kappaSeq = []
    for timingIndex in range(timingNoiseSeq.shape[0]):
        laser.tg = tau * timingNoiseSeq[timingIndex]
        inf = OptimizerIon.exactInfidelity(ionChip, laser, index)
        kappa = QIonEvolution.twoIonCoupling(ionChip, laser, index)
        infidelitySeq.append(inf)
        kappaSeq.append(kappa)
        laser.tg = tau

    return [[timingNoiseSeq, infidelitySeq], [timingNoiseSeq, kappaSeq]]


def noiseFeature(ionChip: QChain1D, laser: QLaser, indexIon: list, noise: float, timeNoise: float):
    r"""
    Plot function to show many features except the phonon-ion coupling, including the waveform of the laser Rabi
    frequency sequence and the phase sequence, the infidelity performance under the dephasing noise and timing
    noise, the variation of the ion-ion coupling strength $\kappa$ as time goes and under the dephasing noise.

    :param timeNoise: specific timing noise
    :param indexIon: optically interacted ions
    :param ionChip: ion chip class
    :param laser: laser class
    :param noise: specific detuning noise

    :return: figure
    """

    # extract parameters for calculating
    eta = np.array([ionChip.lambDickeAxial[i] * laser.waveVector for i in indexIon])
    axialFre = ionChip.ionAxialFre
    laserSeq = np.array([laser.seq[i] for i in indexIon])
    detuning = laser.detuning
    tau = laser.tg

    # plot noise features
    fig, axs = plt.subplots(2, 4, figsize=(18, 6))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    kappaValueSeq = twoIonCouplingSeq(eta, axialFre, laserSeq, detuning, tau, 10)
    axs[0, 0].plot(np.array(range(len(kappaValueSeq))) * (1e6 * tau / (len(kappaValueSeq))), kappaValueSeq)
    axs[0, 0].set_title(r"Ion-ion coupling dynamics")
    axs[0, 0].set_xlabel(r"Time ($\mu$s)")
    axs[0, 0].set_ylabel(r"Ion-ion coupling strength $\chi$")
    axs[0, 1].bar(np.array(range(len(laserSeq[0][0]))) * (1e6 * tau / len(laserSeq[0][0])),
                  laserSeq[0][0] / 1e6, width=12)
    axs[0, 1].set_title('Laser pulse %d modulus' % indexIon[0])
    axs[0, 1].set_xlabel(r"Time ($\mu$s)")
    axs[0, 1].set_ylabel(r"Rabi frequency $\Omega$ (MHz)")
    axs[0, 2].bar(np.array(range(len(laserSeq[0][0]))) * (1e6 * tau / len(laserSeq[0][0])), laserSeq[0][1], width=12)
    axs[0, 2].set_title('Laser pulse %d phase' % indexIon[0])
    axs[0, 2].set_xlabel(r"Time ($\mu$s)")
    axs[0, 2].set_ylabel("Laser phase")
    infidelityPhaseSeq, kappaPhaseSeq = dephasingNoise(ionChip, laser, indexIon, noise)
    axs[0, 3].plot(kappaPhaseSeq[0], kappaPhaseSeq[1])
    axs[0, 3].set_title(r'Coupling $\chi$ under dephasing noise')
    axs[1, 0].set_xlabel(r"Laser detuning drift $\delta$ (kHz)")
    axs[0, 3].set_ylabel(r"Ion-ion coupling strength $\chi$")
    axs[1, 0].plot(infidelityPhaseSeq[0], infidelityPhaseSeq[1])
    axs[1, 0].set_title('Dephasing noise')
    axs[1, 0].set_xlabel(r"Laser detuning drift $\delta$ (kHz)")
    axs[1, 0].set_ylabel("Infidelity")
    axs[1, 1].bar(np.array(range(len(laserSeq[1][0]))) * (1e6 * tau / len(laserSeq[0][0])),
                  laserSeq[1][0] / 1e6, width=12)
    axs[1, 1].set_title('Laser pulse %d modulus' % indexIon[1])
    axs[1, 1].set_xlabel(r"Time ($\mu$s)")
    axs[1, 1].set_ylabel(r"Rabi frequency $\Omega$ (MHz)")
    axs[1, 2].bar(np.array(range(len(laserSeq[1][0]))) * (1e6 * tau / len(laserSeq[0][0])), laserSeq[1][1], width=12)
    axs[1, 2].set_title('Laser pulse %d phase' % indexIon[1])
    axs[1, 2].set_xlabel(r"Time ($\mu$s)")
    axs[1, 2].set_ylabel("Laser phase")

    infidelityTimSeq, kappaTimSeq = timingNoise(ionChip, laser, indexIon, timeNoise)
    axs[1, 3].plot(infidelityTimSeq[0], infidelityTimSeq[1])
    axs[1, 3].set_title('Timing noise')
    axs[1, 3].set_xlabel("Pulse timing scale factor")
    axs[1, 3].set_ylabel("Infidelity")
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    plt.show()
