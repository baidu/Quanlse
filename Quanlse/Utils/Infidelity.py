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
Infidelity tools
"""

from numpy import all, ndarray, conjugate, transpose, shape, trace, dot, isclose, allclose
from scipy import linalg
from math import pow

from Quanlse.QPlatform.Error import ArgumentError
from Quanlse.QOperator import dagger
from Quanlse.Utils.Functions import project


def unitaryInfidelity(uGoal: ndarray, uReal: ndarray, subSysNum: int) -> float:
    r"""
    Return the unitary infidelity of the ``uReal`` with ``uGoal``.
    The unitary infidelity is calculated using trace distance:

    :math:`g = 1 - \frac{1}{{\rm dim}(U)} \left| {\rm Tr} \left( U_{\rm goal}^{\dagger} U_{\rm real} \right) \right|.`

    :param uGoal: target unitary
    :param uReal: real unitary
    :param subSysNum: number of qubits
    :return: value of unitary infidelity
    """
    dimUGoal = shape(uGoal)[0]
    dimUReal = shape(uReal)[0]
    # Calculate qubit levels
    levelUGoal = int(pow(dimUGoal, 1 / subSysNum))
    levelUReal = int(pow(dimUReal, 1 / subSysNum))
    # We first check whether the dimension of the computational state is reasonable
    if levelUGoal == levelUReal:
        uGoal = transpose(conjugate(uGoal))
        return 1 - abs(trace(dot(uGoal, uReal))) / dimUGoal
    elif levelUGoal > levelUReal:
        uGoalProj = project(uGoal, subSysNum, levelUGoal, levelUReal)
        uGoalProj = transpose(conjugate(uGoalProj))
        return 1 - abs(trace(dot(uGoalProj, uReal))) / levelUReal
    else:
        uGoal = transpose(conjugate(uGoal))
        uRealProj = project(uReal, subSysNum, levelUReal, levelUGoal)
        return 1 - abs(trace(dot(uGoal, uRealProj))) / dimUGoal


def isRho(rho: ndarray, tol: float = 1e-7):
    r"""
    Check if the input matrix satisfies the following three requirements: Non-negative
    definite, Hermitian, Normalized.

    :param rho: the input matrix
    :param tol: the tolerance of the imprecision
    :return: -1 if not Hermitian; -2 if not normalized; -3 if not non-negative definite,
        1 when it satisfies the conditions of density operator.
    """

    # Check if rho is Hermitian
    if not allclose(rho - conjugate(transpose(rho)), 0.):
        return -1

    # Check if rho is normalized
    if not isclose(trace(rho), 1):
        return -2

    # Check if rho is non-negative definite
    if not all(linalg.eigvalsh(rho) > -tol):
        return -3

    return 1


def rhoInfidelity(rhoGoal: ndarray, rhoReal: ndarray) -> float:
    r"""
    Calculate the infidelity of the two density matrices.

   :param rhoGoal: the target final density matrix
   :param rhoReal: the real final density matrix
   :return: calculated infidelity of the input two density matrices
   """

    if rhoGoal.shape != rhoReal.shape:
        raise ArgumentError("The dimensions of two the input matrices are not matched.")

    rhoGoalSqrt = linalg.sqrtm(rhoGoal)

    return 1 - trace(linalg.sqrtm(rhoGoalSqrt @ rhoReal @ rhoGoalSqrt)) ** 2


def traceDistance(rhoGoal: ndarray, rhoReal: ndarray) -> float:
    r"""
    Calculate the trace distance between two density matrices.

    :param rhoGoal: the target final density matrix
    :param rhoReal: the real final density matrix
    :return: the trace distance of the two input two density matrices
    """

    if rhoGoal.shape != rhoReal.shape:
        raise ArgumentError("The dimensions of two the input matrices are not matched.")

    if not isRho(rhoGoal) or not isRho(rhoReal):
        raise ArgumentError("The input matrix doesn't meet the criteria of density matrix.")

    return 0.5 * trace(linalg.sqrtm(dagger(rhoGoal - rhoReal) @ (rhoGoal - rhoReal)))
