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
Benchmark
"""

from typing import Dict, Any, List
from Quanlse.Utils.Tools import project
from Quanlse.Utils import Operator
from Quanlse.Utils import Hamiltonian
from Quanlse.Utils import Tools
import numpy


def evolution(ham: Dict[str, Any], stateInitial: numpy.ndarray = None, matrix: numpy.ndarray = None) -> Dict[str, Any]:
    """
    Return the expectation value of the given matrix for given initial states. The input can be a list
    containing all the matrices and states that the users want to calculate.

    :param ham: Hamiltonian Dictionary
    :param stateInitial: the list of numpy.ndarray that represents an initial state in the Hilbert space
    :param matrix: the list of matrix whose expectation values to be evaluated
    :return: the dictionary that contains all the expectation values for all initial states
    """

    # If the input data is not list, we transform it into a list
    if type(matrix) is numpy.ndarray:
        matrixList = [matrix]
    else:
        matrixList = matrix
    if type(stateInitial) is numpy.ndarray:
        stateInitial = [stateInitial]

    result = Hamiltonian.simulate(ham, recordEvolution=True)
    unitaryList = result['evolution_history']
    # x = numpy.linspace(0, ham['circuit']['max_time_ns'], ham['circuit']['max_time_dt'])

    stateList = {}
    # We first calculate all the intermediate states in the process
    for stateIndex, stateItem in enumerate(stateInitial):
        stateListTemp = evolutionList(stateItem, unitaryList)
        stateList[str(stateIndex)] = {}
        stateList[str(stateIndex)]['state_form'] = stateItem
        stateList[str(stateIndex)]['state_evolution_history'] = stateListTemp
        matrixTempDict = {}
        for matrixIndex, matrixItem in enumerate(matrixList):
            matrixTempDict['matrix-' + str(matrixIndex) + '-form'] = matrixItem
            valueList = []
            for stateItemTemp in stateListTemp:
                valueList.append(Tools.expect(matrixItem, stateItemTemp))
            matrixTempDict['matrix-' + str(matrixIndex) + '-value'] = valueList
        stateList[str(stateIndex)]['result'] = matrixTempDict
    return stateList


def evolutionList(state: numpy.ndarray, unitaryList: List) -> list:
    """
    Return the intermediate states with given initial states and the list of unitary matrices.

    :param state: the numpy.ndarray representing the initial state
    :param unitaryList: the list containing different unitary matrices at different times
    :return stateList: the list of intermediate states fot the given unitary matrices
    """
    stateList = []
    for unitaryItem in unitaryList:
        stateList.append(numpy.dot(unitaryItem, state))
    return stateList


def stateTruthTable(unitary, qubitNum, sysLevel, initialBasisList=None, finalBasisList=None) -> numpy.ndarray:
    """
    Generate the truth table of a quantum gate contains the probability of the system being in each
    possible basis states at the end of an operation for each possible initial state.

    :param unitary: the unitary matrix
    :param qubitNum: the number of qubits
    :param sysLevel: the energy level of the system
    :param initialBasisList: the list initial basis
    :param finalBasisList: the list final basis
    :return: the population matrix
    """

    resultMatrix = []
    if initialBasisList is None:
        initialBasisList = range(sysLevel ** qubitNum)
    if finalBasisList is None:
        finalBasisList = range(sysLevel ** qubitNum)
    for state in initialBasisList:
        stateVec = Operator.basis(sysLevel ** qubitNum, state)
        stateFinalTemp = unitary @ stateVec
        probVec = []
        for index, item in enumerate(stateFinalTemp):
            if index in finalBasisList:
                probVec.append(abs(item) ** 2)
        resultMatrix.append(probVec)

    return numpy.array(resultMatrix)
