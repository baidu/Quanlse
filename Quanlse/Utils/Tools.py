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
Tools
"""

import numpy
import math
from typing import List, Union


def project(matrix: numpy.ndarray, qubitNum: int, sysLevel: Union[int, List[int]], toLevel: int) -> numpy.ndarray:
    """
    Project a :math:`d`-level (:math:`d` is an integer) multi-qubit matrix to a lower dimension.

    :param matrix: uReal in ``sysLevel``-dimensional Hilbert space
    :param qubitNum: number of qubits
    :param sysLevel: the energy level of input matrix
    :param toLevel: the target energy level
    :return: uReal in ``toLevel``-dimensional Hilbert space
    """
    if isinstance(sysLevel, int):
        assert toLevel < sysLevel, "The target level should be less than current level."
        # Initialization
        tmpM = numpy.zeros((sysLevel, sysLevel), dtype=int)
            
        # Construct the single qubit matrix.
        for d1 in range(toLevel):
            for d2 in range(toLevel):
                tmpM[d1, d2] = 1
        # Construct the tensor product matrix.
        kronMat = numpy.array([1], dtype=int)
        for _ in range(qubitNum):
            kronMat = numpy.kron(kronMat, tmpM)
        # Output the projected matrix.
        newMat = numpy.zeros((toLevel ** qubitNum, toLevel ** qubitNum), dtype=complex)
        toX, toY = 0, 0
        for x in range(sysLevel ** qubitNum):
            dropLine = True
            for y in range(sysLevel ** qubitNum):
                if kronMat[x, y] == 1:
                    dropLine = False
                    newMat[toX, toY] = matrix[x, y]
                    toY += 1
            toY = 0
            if dropLine is False:
                toX += 1
    if isinstance(sysLevel, list):
        assert toLevel < min(sysLevel), "The target level should be less than the minimum level of one of the qubit."
        # Construct the tensor product matrix for this system 
        kronMat = numpy.array([1], dtype=int)
        for level in sysLevel:
            tmpM = numpy.zeros((level, level), dtype=int)
            for d1 in range(toLevel):
                for d2 in range(toLevel):
                    tmpM[d1, d2] = 1
            kronMat = numpy.kron(kronMat, tmpM)
        # Initialize the output matrix 
        newMat = numpy.zeros((toLevel ** qubitNum, toLevel ** qubitNum), dtype=complex)
        dim = matrix.shape[0]
        toX, toY = 0, 0
        for x in range(dim):
            dropLine = True
            for y in range(dim):
                if kronMat[x, y] == 1:
                    dropLine = False
                    newMat[toX, toY] = matrix[x, y]
                    toY += 1
            toY = 0
            if dropLine is False:
                toX += 1

    # Return the output matrix
    return newMat


def unitaryInfidelity(uGoal: numpy.ndarray, uReal: numpy.ndarray, qubitNum: int) -> float:
    r"""
    Return the unitary infidelity of the ``uReal`` with ``uGoal``.
    The unitary infidelity is calculated using trace distance:

    :math:`g = 1 - \frac{1}{{\rm dim}(U)} \left| {\rm Tr} \left( U_{\rm goal}^{\dagger} U_{\rm real} \right) \right|.`

    :param uGoal: target unitary
    :param uReal: real unitary
    :param qubitNum: number of qubits
    :return: value of unitary infidelity
    """
    dimUGoal = numpy.shape(uGoal)[0]
    dimUReal = numpy.shape(uReal)[0]
    # Calculate qubit levels
    levelUGoal = int(math.pow(dimUGoal, 1 / qubitNum))
    levelUReal = int(math.pow(dimUReal, 1 / qubitNum))
    # We first check whether the dimension of the computational state is reasonable
    if levelUGoal == levelUReal:
        uGoal = numpy.transpose(numpy.conjugate(uGoal))
        return 1 - abs(numpy.trace(numpy.dot(uGoal, uReal))) / dimUGoal
    elif levelUGoal > levelUReal:
        uGoalProj = project(uGoal, qubitNum, levelUGoal, levelUReal)
        uGoalProj = numpy.transpose(numpy.conjugate(uGoalProj))
        return 1 - abs(numpy.trace(numpy.dot(uGoalProj, uReal))) / levelUReal
    else:
        uGoal = numpy.transpose(numpy.conjugate(uGoal))
        uRealProj = project(uReal, qubitNum, levelUReal, levelUGoal)
        return 1 - abs(numpy.trace(numpy.dot(uGoal, uRealProj))) / dimUGoal


def computationalBasisList(qubitNum: int, sysLevel: int) -> List[str]:
    """
    Return a list of strings labeling eigenstates.
    For example, ``computationalBasisList(2, 3)`` will return:
    ``['00', '01', '02', '10', '11', '12', '20', '21', '22']``

    :param qubitNum: the number of qubits in the system
    :param sysLevel: the energy level of the qubits in the system
    :return: the list of strings labeling eigenstates
    """
    assert isinstance(sysLevel, int), 'The system level can only be an integer.'
    itemCount = sysLevel ** qubitNum
    strList = []
    for index in range(itemCount):
        bStr = ''
        for qu in range(qubitNum):
            bStr = f"{int(index / sysLevel ** qu) % sysLevel}{bStr}"
        strList.append(bStr)
    return strList


def generateBasisIndexList(basisStrList: List[str], sysLevel: int) -> List[int]:
    """
    Return a list of integers which indicates the basis indices according to the input basis string list.
    For example, ``generateBasisIndexList(['00', '01', '10', '11'], 3)`` will return:
    ``[0, 1, 3, 4]``

    :param basisStrList: basis string list
    :param sysLevel: the energy level of qubits in the system.
    :return: basis indices list
    """
    assert isinstance(sysLevel, int), 'The system level can only be an integer.'
    strLen = [len(item) for item in basisStrList]
    assert max(strLen) == min(strLen), "All input digital strings should have same length."
    digLen = max(strLen)

    def translateStrToInt(strN: str) -> int:
        """ Translate a string to int """
        intNum = 0
        for digIndex, charN in enumerate(strN):
            dig = int(charN)
            assert dig < sysLevel, f"Digit '{dig}' is greater than sysLevel '{sysLevel}'."
            intNum += (sysLevel ** (digLen - digIndex - 1)) * dig
        return intNum

    basisIntList = []
    for strNum in basisStrList:
        basisIntList.append(translateStrToInt(strNum))

    return basisIntList


def expect(matrix: numpy.ndarray, state: numpy.ndarray) -> float:
    """
    Return the expectation value of the matrix in the given state.

    :param matrix: the given matrix
    :param state: the given state
    :return: expectation value of matrix in given state
    """

    def isSquare(m: numpy.ndarray) -> bool:
        """
        This function is used to check whether the matrix is a square matrix
        """
        return all(len(row) == len(m) for row in m)

    # We firstly need to check whether the matrix is square
    assert isSquare(matrix), 'Matrix is not a square matrix'
    assert state.shape[0] == matrix.shape[1], 'Dimension Mismatch'
    stateDagger = numpy.conjugate(numpy.transpose(state))
    stateMatrix = numpy.dot(state, stateDagger)
    expectValue = numpy.trace(numpy.dot(matrix, stateMatrix))
    # We hope to extract the only element in the numpy.trace result
    if type(expectValue) is numpy.ndarray:
        expectValue = numpy.reshape(expectValue, 1, )[0]
    return expectValue


def tensor(matrixList: list) -> numpy.ndarray:
    """
    Return the tensor product of all the matrices in the list.

    :param matrixList: the list of the matrices to take the tensor product
    :return: tensor product of the matrices in the list
    """
    # We firstly need to check if all the matrix in the list a numpy.ndarray
    assert all(type(matrixItem) is numpy.ndarray for matrixItem in matrixList), \
        'Type of the matrix in the list is not consistent.'
    matrixReturn = numpy.array([[1.0]], dtype=complex)
    for matrixIndex, matrixItem in enumerate(matrixList):
        matrixReturn = numpy.kron(matrixReturn, matrixItem)
    return matrixReturn
