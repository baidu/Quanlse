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
We show how to optimize the laser pulse sequences in trapped ion quantum computing.
While in trapped ion, because Hilbert is too large and the phonon operation
is complex to describe, we need to get approximate formula of multi-qubit gates
infidelity. And by define the goal function of robust Mølmer-Sørensen gate, we
can find a dephasing and timing noise robust laser control sequences.
"""

import numpy as np
from math import pi
import math
import scipy.optimize as optimize
from Quanlse.TrappedIon.Optimizer.QIonEvolution import phononIonCoupling, phononIonCouplingExact, twoIonCoupling
from Quanlse.TrappedIon.Optimizer.QIonEvolution import getDiff, alphaExactDiff
from Quanlse.TrappedIon.QIonSystem import QChain1D, QLaser


def infidelity(ionChip: QChain1D, laser: QLaser, indexIon: list) -> float:
    """
    Calculate the infidelity of ion gate

    :param ionChip: The ionChip object
    :param laser: The control laser object
    :param indexIon: The index ion position list control by laser
    :return: approximated infidelity
    """

    resAlphaSum = phononIonCoupling(ionChip, laser, indexIon)
    resKappa = twoIonCoupling(ionChip, laser, indexIon)
    resDiff = getDiff(ionChip, laser, indexIon)
    result = (8 * ((pi / 4) - resKappa) ** 2 + resAlphaSum + 0.5 * resDiff) / 8
    return result


def exactInfidelity(ionChip: QChain1D, laser: QLaser, indexIon: list) -> float:
    """
    The infidelity function for MS gate accurately

    :param ionChip: The ionChip object
    :param laser: The control laser object
    :param indexIon: The index ion position list control by laser
    :return: the exact infidelity for pi/4 MS gate
    """

    resAlphaSum = phononIonCouplingExact(ionChip, laser, indexIon)
    chi = twoIonCoupling(ionChip, laser, indexIon)
    resDiff = alphaExactDiff(ionChip, laser, indexIon)
    result = 1 - (2 + 2 * math.sin(2 * chi) * resAlphaSum + resDiff) / 8
    return result


def optimizeInitial(ionChip: QChain1D, laser: QLaser, indexIon: list) -> tuple:
    """
    The initialization function for optimization.

    :param ionChip: The ionChip object
    :param laser: The control laser object
    :param indexIon: The index ion position list control by laser
    :return: initial optimization variables, optimization bounds, phase-modulation parameters
    """
    # initialization factor
    unitFactor = 1e6

    # extract key features
    eta = np.array(ionChip.lambDickeAxial * laser.waveVector)
    frequency = ionChip.ionAxialFre
    detuning = laser.detuning
    tau = laser.tg
    segments = laser.segments
    maxRabi = laser.maxRabi

    # determine whether can reach aiming coupling
    relativeDetuning = np.array([frequency[i] - detuning for i in range(len(frequency))])
    index = np.argmin(abs(relativeDetuning))
    numFreeSegments = int(np.ceil(segments / 2))
    rabiLess = abs(pi / (2 * eta[indexIon[0]][index] * eta[indexIon[1]][index])) ** 0.5
    if maxRabi < rabiLess:
        from Quanlse.QPlatform import Error
        raise Error.ArgumentError("The maxRabi rate is too low")

    # generate initial optimization values
    omegaInitial = abs(
        relativeDetuning[index] / (tau * eta[indexIon[0]][index] * eta[indexIon[1]][index])) ** 0.5 * np.ones(
        numFreeSegments) / unitFactor
    phiInitial = np.array([(-1) ** i for i in range(numFreeSegments)])
    x0 = np.array([[omegaInitial, phiInitial], [omegaInitial, -phiInitial]]).reshape(2 * 2 * numFreeSegments)

    # generate corresponding bounds for optimization
    bounds = []
    for i in range(2):
        for j in range(numFreeSegments):
            bounds.append([0, maxRabi / unitFactor])
        for j in range(numFreeSegments):
            bounds.append([-pi, pi])

    # parameter for phase modulation
    phaseAdd = -relativeDetuning[index] * tau / 2 + pi
    return x0, bounds, -phaseAdd


def costFunc(ionChip: QChain1D, laser: QLaser, indexIon: list, noise: float) -> float:
    """
    The cost function of optimization

    :param ionChip: The ionChip object
    :param laser: The control laser object
    :param indexIon: The index ion position list control by laser
    :param noise: scope of noise
    :return: cost value
    """
    a, b, c = [4, 0.5, 400]  # cost function parameters

    # get noiseless approximated infidelity
    detuning = laser.detuning
    resAlpha = phononIonCoupling(ionChip, laser, indexIon)
    resKappa = twoIonCoupling(ionChip, laser, indexIon)
    resDiff = getDiff(ionChip, laser, indexIon)

    # get noise plus approximated infidelity
    laser.detuning = detuning + noise
    resAlphaSum = phononIonCoupling(ionChip, laser, indexIon)
    resKappaSum = twoIonCoupling(ionChip, laser, indexIon)
    resDiffSum = getDiff(ionChip, laser, indexIon)

    # get noise minus approximated infidelity
    laser.detuning = detuning - noise
    resAlphaMinus = phononIonCoupling(ionChip, laser, indexIon)
    resKappaMinus = twoIonCoupling(ionChip, laser, indexIon)
    resDiffMinus = getDiff(ionChip, laser, indexIon)

    # add them up to form cost values
    laser.detuning = detuning
    result = (a * ((pi / 4) - resKappa) ** 2 + resAlpha + b * resDiff) * c
    resultSum = (a * ((pi / 4) - resKappaSum) ** 2 + resAlphaSum + b * resDiffSum) * c
    resultMinus = (a * ((pi / 4) - resKappaMinus) ** 2 + resAlphaMinus + b * resDiffMinus) * c
    result = 10 * result + (resultSum + resultMinus)
    return result


def optimizeIonSymmetry(ionChip: QChain1D, laser: QLaser, indexIon: list, noise: float) -> QLaser:
    """
    The optimization function that optimize the laser pulse in symmetry method

    :param ionChip: The ionChip object
    :param laser: The control laser object
    :param indexIon: The index ion position list control by laser
    :param noise: scope of noise
    :return: optimized laser pulse sequence (omega-symmetry and phase-antisymmetry)
    """

    # initialization
    x0, bounds, phase = optimizeInitial(ionChip, laser, indexIon)

    # normalize the units into μs and MHz for optimization
    normal = 1e6
    ionChip.ionAxialFre = ionChip.ionAxialFre / normal
    laser.detuning = laser.detuning / normal
    laser.maxRabi = laser.maxRabi / normal
    laser.tg = laser.tg * normal
    noise = noise / normal

    def goalFun(x):
        """
        Calculate cost function
        """
        laser.seq = laser.symmetrySeq(x=x, ionNumber=ionChip.ionNumber, indexIon=indexIon)
        result = costFunc(ionChip, laser, indexIon, noise)
        return result

    opt = np.array(optimize.fmin_slsqp(func=goalFun, x0=x0, bounds=bounds, iter=100))
    laser.seq = laser.symmetrySeq(x=opt, ionNumber=ionChip.ionNumber, indexIon=indexIon)
    print("Two ion coupling opt:", twoIonCoupling(ionChip, laser, indexIon))
    print("Infidelity:", infidelity(ionChip, laser, indexIon))

    # back normalization
    ionChip.ionAxialFre = ionChip.ionAxialFre * normal
    for i in range(len(laser.seq)):
        laser.seq[i][0] = laser.seq[i][0] * normal
    laser.detuning = laser.detuning * normal
    laser.maxRabi = laser.maxRabi * normal
    laser.tg = laser.tg / normal
    return laser
