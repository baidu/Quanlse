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
Utils functions
"""
import copy
from math import atan2, pi
from typing import List, Union, Iterable, Optional, Callable
import itertools
from itertools import product

import numpy
from numpy import all, array, ndarray, zeros, kron, trace, dot, eye, angle, real, exp, \
    cos, arange, fft, argmax, abs, expand_dims, block, identity, meshgrid, sqrt, arctan, conj, prod

from scipy.optimize import fmin
from scipy import linalg
from scipy.special import laguerre, genlaguerre
from math import factorial

from Quanlse.QPlatform.Error import ArgumentError
from Quanlse.QOperator import QOperator, dagger, destroy
from Quanlse.Utils.Plot import plotPop


def generateOperatorKey(subSysNum: int, operator: Union[QOperator, List[QOperator]]) -> str:
    """
    Generate key for operators.

    :param subSysNum: subsystem's number
    :param operator: the corresponding operator object(s)
    :return: returned key
    """
    subSysList = []
    nameList = []
    if isinstance(operator, list):
        for op in operator:
            subSysList.append(op.onSubSys)
            nameList.append(op.name)
    else:
        subSysList.append(operator.onSubSys)
        nameList.append(operator.name)

    keyStr = ""
    forward = 0
    for idx in range(subSysNum):
        if idx in subSysList:
            keyStr = keyStr + chr(8855) + f"{nameList[forward]}{idx}"
            forward += 1

    return keyStr.strip(chr(8855))


def combineOperatorAndOnSubSys(subSysNum: int, operators: Union[QOperator, List[QOperator]] = None,
                               onSubSys: Union[int, List[int]] = None) -> Union[QOperator, List[QOperator]]:
    """
    Set onSubSys information into the operator.

    :param subSysNum: subsystem's number
    :param operators: the corresponding operator object(s)
    :param onSubSys: subsystem's index(es)
    :return: returned QOperator object(s)

    """
    # Verify if operator and onSubSys have the same size
    if isinstance(operators, list) and not isinstance(onSubSys, list):
        raise ArgumentError("operator is a list, however onSubSys is not a list.")
    if not isinstance(operators, list) and isinstance(onSubSys, list):
        raise ArgumentError("onSubSys is a list, however operator is not a list.")

    if isinstance(operators, list) and isinstance(onSubSys, list):
        if len(operators) != len(onSubSys):
            raise ArgumentError(f"The size of operator ({len(operators)}) != that "
                                      f"of onSubSys ({len(onSubSys)})!")

    # Verify the range and set the onSubSys
    if isinstance(onSubSys, int):
        if onSubSys >= subSysNum:
            raise ArgumentError(f"onSubSys ({onSubSys}) is larger than the "
                                      f"subSysNum {subSysNum}.")
        operators.onSubSys = onSubSys
        operatorForSave = operators
    elif isinstance(onSubSys, list):
        if max(onSubSys) >= subSysNum:
            raise ArgumentError(f"onSubSys ({onSubSys}) is larger than the "
                                      f"subSysNum {subSysNum}.")
        # Sort the operators according to on onSubSys
        sortedIndex = list(array(onSubSys).argsort())
        sortedOperators = [operators[i] for i in sortedIndex]
        sortedOnSubSys = [onSubSys[i] for i in sortedIndex]

        # Set the onSubSys
        for i in range(len(sortedIndex)):
            sortedOperators[i].onSubSys = sortedOnSubSys[i]
        operatorForSave = sortedOperators
    else:
        raise ArgumentError("onSubSys should be an integer or a list!")

    return operatorForSave


def formatOperatorInput(operators: Union[QOperator, Callable, List[QOperator], List[Callable]],
                        onSubSys: Union[int, List[int]], sysLevel: Union[int, List[int]]) -> Union:
    """
    We allow input the Callable QOperator instances or the function.
    """
    _operators = []
    if isinstance(operators, list):
        for opIdx, op in enumerate(operators):
            if isinstance(op, QOperator):
                # Input operator is an QOperator instance
                _operators.append(copy.deepcopy(op))
            else:
                # Input operator is callable
                if isinstance(sysLevel, list):
                    _operators.append(op(sysLevel[onSubSys[opIdx]]))
                else:
                    _operators.append(op(sysLevel))
    else:
        if isinstance(operators, QOperator):
            # Input operator is an QOperator instance
            _operators = copy.deepcopy(operators)
        else:
            # Input operator is callable
            if isinstance(sysLevel, list):
                _operators = operators(sysLevel[onSubSys])
            else:
                _operators = operators(sysLevel)
    return _operators


def project(matrix: ndarray, qubitNum: int, sysLevel: Union[int, List[int]], toLevel: int) -> ndarray:
    """
    Project a :math:`d`-level (:math:`d` is an integer) multi-qubit matrix to a lower dimension.

    :param matrix: uReal in ``sysLevel``-dimensional Hilbert space
    :param qubitNum: number of qubits
    :param sysLevel: the energy level of input matrix
    :param toLevel: the target energy level
    :return: uReal in ``toLevel``-dimensional Hilbert space
    """
    if len(matrix.shape) == 1 or min(matrix.shape) == 1:
        isMatrix = False
    else:
        isMatrix = True
    newMat = None
    if isinstance(sysLevel, int):
        if toLevel >= sysLevel:
            raise ArgumentError("The target level should be less than the current level.")
        # Initialization
        if isMatrix:
            tmpM = zeros((sysLevel, sysLevel), dtype=int)
        else:
            tmpM = zeros((sysLevel,), dtype=int)
        # Construct the single qubit matrix.
        if isMatrix:
            for d1 in range(toLevel):
                for d2 in range(toLevel):
                    tmpM[d1, d2] = 1
        else:
            for d1 in range(toLevel):
                tmpM[d1] = 1
        # Construct the tensor product matrix.
        kronMat = array([1], dtype=int) if isMatrix else 1
        for _ in range(qubitNum):
            kronMat = kron(kronMat, tmpM)
        # Output the projected matrix.
        if isMatrix:
            newMat = zeros((toLevel ** qubitNum, toLevel ** qubitNum), dtype=complex)
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
        else:
            newMat = zeros((toLevel ** qubitNum,), dtype=complex)
            toX = 0
            for x in range(sysLevel ** qubitNum):
                if kronMat[x] == 1:
                    newMat[toX] = matrix[x]
                    toX += 1
    if isinstance(sysLevel, list):
        if toLevel >= min(sysLevel):
            raise ArgumentError("The target level should be less than the minimum level of one of the qubit.")
        # Construct the tensor product matrix for this system
        kronMat = array([1], dtype=int) if isMatrix else 1
        for level in sysLevel:
            tmpM = zeros((level, level), dtype=int)
            for d1 in range(toLevel):
                for d2 in range(toLevel):
                    tmpM[d1, d2] = 1
            kronMat = kron(kronMat, tmpM)
        # Initialize the output matrix
        newMat = zeros((toLevel ** qubitNum, toLevel ** qubitNum), dtype=complex)
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


def tensor(*args) -> ndarray:
    """
    Return the tensor product of all matrices in the list.

    :param matrixList: the list of matrices to take the tensor product
    :return: tensor product of matrices in the list
    """
    # We firstly need to check if all the matrix in the list a numpy.ndarray
    if len(args) == 1 and isinstance(args[0], List):
        args = tuple(args[0])
    else:
        if isinstance(args, ndarray):
            return args

    matrixReturn = array([[1.0]], dtype=complex)
    for matrixIndex, matrixItem in enumerate(args):
        matrixReturn = kron(matrixReturn, matrixItem)
    return matrixReturn


def basis(d: int, state: int) -> ndarray:
    r"""
    Return the vector of state operator:
    :math:`|0\rangle, |1\rangle, \cdots`.

    :param d: the dimension of the Hilbert space
    :param state: the index of the state
    :return: matrix form of the state operator
    """
    if d <= state:
        raise ArgumentError("The input state index exceeds system dimension")
    matrix = zeros((d, 1), dtype=complex)
    matrix[state] = 1.0
    return matrix


def projector(a: ndarray, b: ndarray = None) -> ndarray:
    r"""
    Return the matrix form of a state: :math:`|a\rangle \langle b|`.

    :param a: ket operator
    :param b: bra operator
    :return: the outer product of the two operators :math:`|a\rangle` and :math:`(|b\rangle)^{\dagger}`
    """
    if b is None:
        returnMatrix = dot(a, dagger(a))
    else:
        returnMatrix = dot(a, dagger(b))
    return returnMatrix


def expect(matrix: ndarray, state: ndarray) -> float:
    """
    Return the expectation value of the matrix in the given state.

    :param matrix: the given matrix
    :param state: the given state (1-d state or 2-d density matrix)
    :return: expectation value of the matrix in given state
    """

    def isSquare(m: ndarray) -> bool:
        """
        This function is used to check whether the matrix is a square matrix
        :param m: input matrix
        :return: a bool value
        """
        return all(len(row) == len(m) for row in m)

    # We firstly need to check whether the matrix is square
    if not isSquare(matrix):
        raise ArgumentError('Matrix is not a square matrix')
    if state.shape[0] != matrix.shape[1]:
        raise ArgumentError('Dimension Mismatch')

    # is the input state ket or a density matrix?
    if state.shape[0] != state.shape[1]:
        # the input state is a pure state
        state = state @ state.conj().T
        expectValue = real(trace(dot(matrix, state)))

    else:
        # the input state is a density operator
        expectValue = real(trace(dot(matrix, state)))

    return expectValue


def computationalBasisList(qubitNum: int, sysLevel: int) -> List[str]:
    """
    Return a list of strings labeling eigenstates.
    For example, ``computationalBasisList(2, 3)`` will return:
    ``['00', '01', '02', '10', '11', '12', '20', '21', '22']``

    :param qubitNum: the number of qubits in the system
    :param sysLevel: the energy level of the qubits in the system
    :return: the list of strings labeling eigenstates
    """
    if not isinstance(sysLevel, int):
        raise ArgumentError('The system level can only be an integer.')
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
    :return: basis indixes list
    """
    if not isinstance(sysLevel, int):
        raise ArgumentError('The system level can only be an integer.')
        
    strLen = [len(item) for item in basisStrList]
    if max(strLen) != min(strLen):
        raise ArgumentError('All input digital strings should have same length.')
    digLen = max(strLen)

    def translateStrToInt(strN: str) -> int:
        """
        Translate a string to int.
        :param strN: input string
        :return: an int value
        """
        intNum = 0
        for digIndex, charN in enumerate(strN):
            dig = int(charN)
            if dig >= sysLevel:
                raise ArgumentError(f"Digit '{dig}' is greater than sysLevel '{sysLevel}'.")
            intNum += (sysLevel ** (digLen - digIndex - 1)) * dig
        return intNum

    basisIntList = []
    for strNum in basisStrList:
        basisIntList.append(translateStrToInt(strNum))

    return basisIntList


def partialTrace(rho: ndarray, subNum: int, dimList: List[int], index: Union[int, List[int]], mode=0) -> ndarray:
    """
    Partial trace a :math:`d`-level (:math:`d` is an integer) multi-qubit matrix to a lower dimension.

    :param rho: global density matrix to be partially traced
    :param subNum: number of qubits
    :param dimList: the dimension of each subsystem
    :param index: index of subsystem(s) (to be reserved), starts from 1
    :param mode: 0(default)--reserve the subsystems in index list, 1--trace the subsystems in index list
    :return: reserved density matrix
    """
    # the complete indexList
    indexFull = [i + 1 for i in range(subNum)]

    # we use dimListCopy to avoid changing dimList
    dimListCopy = dimList[:]

    a = rho.shape
    if a[0] != a[1]:
        raise ArgumentError("The input matrix is invalid")

    def PT(A, Num, DList, idx):
        c = A.shape

        if Num != len(DList):
            raise ArgumentError("Invalid dimList: dimList is inconsistent with subNum")

        dim = 1
        for k in range(len(DList)):
            if DList[k] == 1 or 0:
                raise ArgumentError("Invalid dimList: Subsystem cannot be one or zero dimension")
            else:
                dim *= DList[k]

        if dim != c[0]:
            raise ArgumentError("Invalid dimList: dimList is inconsistent with rho's dimension")

        if idx <= 0 or idx > Num:
            raise ArgumentError("Invalid index")

        # denote the dimension of the index to be traced as d2
        d2 = DList[idx - 1]
        d1 = 1
        d3 = c[0]

        # denote the dimension in front of the index to be traced as d1
        for k in range(idx - 1):
            d1 = d1 * DList[k]
            d3 = d3 // DList[k]

        # denote the dimension behind the index to be traced as d3
        d3 = d3 // d2

        I1 = eye(d1)
        I2 = eye(d2)

        if d3 != 0:
            I3 = eye(d3)
        else:
            I3 = 1
            d3 = 1
        d = d1 * d3

        # initialize the traced state
        subState = zeros((d, d), dtype=complex)

        # the process of partial trace
        for i in range(d2):
            basis_ = kron(kron(I1, I2[:, i]), I3)
            subState += dot(dot(basis_, A), basis_.T)

        return subState

    if isinstance(index, list):
        index.sort()
        for i in range(len(index)):
            if index[i] == 0:
                raise ArgumentError("Invalid index list: index starts from 1")

        for i in range(len(index)):
            for j in range(len(index)):
                if index[i] == index[j] and i != j:
                    raise ArgumentError("Invalid index list: repeated index")

        if len(index) > subNum:
            raise ArgumentError("Invalid index list: len(index) > subNum")

        if set(index).issubset(indexFull) is False:
            raise ArgumentError("Invalid index list")

        if mode == 0:
            index = list(set(indexFull) - set(index))
            mode = 1

        if mode == 1:
            for i in range(len(index)):
                rho = PT(rho, subNum, dimListCopy, index[i])

                # partition new subsystems, we use dimListCopy to avoid changing dimList from list.pop
                dimListCopy.pop(index[i] - 1)
                subNum = subNum - 1

                # reset the index for next partial trace
                index = array(index) - 1
                index.tolist()
        else:
            raise ArgumentError("Invalid mode")

    elif mode == 1:
        rho = PT(rho, subNum, dimList, index)

    elif mode == 0:
        index = list(set(indexFull) - {index})

        for i in range(len(index)):
            rho = PT(rho, subNum, dimListCopy, index[i])

            # partition new subsystems
            dimListCopy.pop(index[i] - 1)

            subNum = subNum - 1
            # reset the index for next partial trace
            index = array(index) - 1
            index.tolist()

    else:
        raise ArgumentError("Invalid mode")

    return rho


def globalPhase(U: ndarray) -> float:
    r"""
    Compute the global phase of a 2*2 unitary matrix.
    Each 2*2 unitary matrix can be equivalently characterized as:

    :math:`U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)`

    We aim to compute the global phase `\alpha`.
    See also Theorem 4.1 in `Nielsen & Chuang`'s book.

    :param U: the matrix representation of the 2*2 unitary
    :return: the global phase of the unitary matrix
    """
    # Notice that the determinant of the unitary is given by e^{2i\alpha}
    coe = linalg.det(U) ** (-0.5)
    alpha = - angle(coe)
    return alpha


def fromMatrixToAngles(U: ndarray) -> List[float]:
    r"""
    Compute the Euler angles `(\alpha,\theta,\phi,\lambda)` for a single-qubit gate.
    Each single-qubit gate can be equivalently characterized as:

    :math:`U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda) \\
    = e^{i(\alpha-\phi/2-\lambda/2)}
    \begin{bmatrix}
    \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\
    e^{i\phi}\sin(\theta/2) & e^{i(\phi+\lambda)}\cos(\theta/2)
    \end{bmatrix}`

    We aim to compute the parameters `(\alpha, \theta,\phi,\lambda)`.
    See Theorem 4.1 in `Nielsen & Chuang`'s book for details.


    :param U: the matrix representation of the qubit unitary
    :return: the Euler angles in List
    """
    if U.shape != (2, 2):
        raise ArgumentError("in fromMatrixToAngles(): input should be a 2x2 matrix!")
    # Remove the global phase
    alpha = globalPhase(U)
    U = U * exp(- 1j * alpha)
    U = U.round(10)
    # Compute theta
    theta = 2 * atan2(abs(U[1, 0]), abs(U[0, 0]))

    # Compute phi and lambda
    phiplambda = 2 * angle(U[1, 1])
    phimlambda = 2 * angle(U[1, 0])
    phi = (phiplambda + phimlambda) / 2.0
    lam = (phiplambda - phimlambda) / 2.0

    return [alpha, theta, phi, lam]


def population(rho: ndarray, subNum: int, dimList: List[int], plot=False) -> dict:
    """
    Output a dictionary to show population of multi-qubit matrix

    :param rho: density matrix
    :param subNum: number of qubits
    :param dimList: the dimension of each subsystem
    :param plot: an option to plot population
    :return: a dictionary illustrate population of each energy level
    """

    a = rho.shape
    if a[0] != a[1]:
        raise ArgumentError("The input matrix is invalid")

    if subNum != len(dimList):
        raise ArgumentError("Invalid dimList: dimList is inconsistent with subNum")

    dim = 1
    for j in range(len(dimList)):
        if dimList[j] == 1 or 0:
            raise ArgumentError("Invalid dimList: Subsystem cannot be one or zero dimension")
        else:
            dim *= dimList[j]

    if dim != a[0]:
        raise ArgumentError("Invalid dimList: dimList is inconsistent with rho's dimension")

    # generate a complete sequence of population
    maxi = max(dimList)
    l_ = zeros(maxi, dtype=int)
    for i in range(maxi):
        l_[i] = '%d' % i
    comp = array(list(product(l_, repeat=subNum)))

    # valuing invalid population
    for i in range(maxi ** subNum):
        for j in range(subNum):
            if comp[i][j] >= dimList[j]:
                comp[i] = 100

    def is_none(n):
        return n[0] == 100

    # filter invalid population, generating a complete sequence of valid population
    pop_name = itertools.filterfalse(is_none, list(comp))
    list_ = list(pop_name)

    popDict = {}

    P = []
    popList = []
    popName = []
    for k in range(dim):
        P.append(basis(dim, k) @ dagger(basis(dim, k)))

    for k in range(dim):
        atr = ''.join(str(i) for i in list_[k])
        pop = abs(trace(rho @ P[k]))
        popList.append(pop)
        popDict[atr] = pop
        popName.append(atr)

    if plot is True:
        # Draw the population of computational basis
        plotPop(popName, popList, xLabel="Computational Basis", yLabel="Population")

    return popDict


def getPopulationOnQubit(popList: List[float], onSubSys: int, sysLevel: int):
    """
    Get the population on the given qubit.

    :param popList: the population list of the full system
    :param onSubSys: the index of subsystem to obtain the population
    :param sysLevel: the level of the system (just supports int at present)
    """

    if isinstance(sysLevel, int):
        targetQLvl = sysLevel
    else:
        raise ArgumentError("sysLevel must be an integer!")

    popRes = [0. for _ in range(targetQLvl)]

    for targetQubitLvlIdx in range(targetQLvl):
        # Traverse all the basis of the target qubit
        for itemIdx, itemVal in enumerate(popList):
            d, m = divmod(int(itemIdx / (sysLevel ** onSubSys)), targetQLvl)
            if m == targetQubitLvlIdx:
                popRes[targetQubitLvlIdx] += itemVal

    return popRes


def findIndex(referenceVecs: ndarray, indexKet: Union[List[ndarray], ndarray]):
    """
    Find the index of the given kets in the indexKet using inner product.

    :param referenceVecs: ndarray of reference vectors.
    :param indexKet: target ket or kets list.
    :return: list of index number.
    """
    # initialize index list
    idxList = []
    refVecList = copy.deepcopy(referenceVecs.T)

    if isinstance(indexKet, list):

        for ket in indexKet:
            innerProd = [vec for vec in refVecList @ ket]
            innerVal = max(innerProd)
            index = innerProd.index(innerVal)
            idxList.append(index)

        return idxList

    else:
        innerProd = [vec for vec in refVecList @ indexKet]
        innerVal = max(innerProd)
        index = innerProd.index(innerVal)
        return index


def fitCR(xData: array, yData: array):
    """
    Fit the rabi oscillation of cross-resonance effect.

    :param xData: The data of x values.
    :param yData: The data of y values.

    :return: The callable fitting function and the fitting parameters.
    """

    def fit(x, p):
        return 0.5 * cos(2 * pi * p[1] * x) + 0.5

    def error(p, x, y, fitFunc):
        return ((y - fitFunc(x, p)) ** 2).sum() / len(y)

    step = xData[1] - xData[0]
    pows = abs(fft.fft(yData))
    freqs = fft.fftfreq(xData.size, step)
    index = argmax(pows[freqs > 0])

    a1Init = abs(yData).max()
    a2Init = freqs[index + 1]

    paraFit = fmin(error, [a1Init, a2Init], (xData, yData, fit))
    return fit, paraFit


def blockDiag(matrix: ndarray, subIndex: List[int]):
    """
    Block diagonalize a given matrix using the principle of least action.

    :param matrix: The given matrix to be block diagonalized.
    :param subIndex: The indexes of sub-system.

    :return: Block diagonalization Matrix and transform Unitary
    """

    if matrix.shape[0] is not matrix.shape[1]:
        raise ArgumentError('Not a square matrix')

    if len(subIndex) > max(matrix.shape):
        raise ArgumentError(f'Number of indexes {len(subIndex)} exceeds matrix dimension len{matrix}')

    # Calculate the eigenvectors and eigenvalues of matrix.
    valsInit, vecsInit = linalg.eig(matrix)

    # Rearrange eigenvalues and corresponding eigenvectors in the ascending order.
    index = valsInit.argsort()
    vecs = vecsInit[:, index]
    vals = valsInit[index]

    s1 = vecs[0:len(subIndex), subIndex]
    s2 = vecs[len(subIndex):len(matrix), subIndex]
    X = -dagger(s2 @ linalg.inv(s1))

    XDag = X.T.conj()
    mat1 = block([identity(X.shape[0]), X])
    mat2 = block([-XDag, identity(XDag.shape[0])])
    U = block([[mat1], [mat2]])
    T = U @ linalg.sqrtm(linalg.inv(dagger(U) @ U))

    # Compute the block-diagonal matrix
    matBD = dagger(T) @ matrix @ T

    return matBD, T


def eigenSystem(matrix: ndarray):
    """
    compute the eigenvalues and the corresponding eigenvectors for the given matrix.
    (sorted by eigenvalues: Ascending order)

    :param matrix: base matrix of the eigen system

    :return: sorted eigenvalues (eigenenergies) and corresponding eigenvectors (eigenstates) (ordered by column)
    """
    # Solve eigenvalues problem
    eigenVals, eigenVecs = linalg.eig(matrix)

    # Rearrange eigenvalues and eigenvectors in the ascending order
    sortedIndex = eigenVals.argsort()
    eigenVecs = eigenVecs[:, sortedIndex]
    eigenVals = eigenVals[sortedIndex]
    eigenVecs = eigenVecs

    return eigenVals.real, eigenVecs


def wigner(rho: ndarray, xRange: ndarray, yRange: ndarray):
    """
    Calculate the wigner function of density matrix using laguerre polynomial.

    :param rho: Input density matrix.
    :param xRange: The range of the X quadrature in the phase space.
    :param yRange: The range of the y quadrature in the phase space.
    """
    X, P = meshgrid(xRange, yRange)
    T = X ** 2 + P ** 2

    def _wignerLaguerre(_m, _n):
        """
        Return wigner function for state |n><m|
        """
        if _m == _n:
            poly = laguerre(_n)
            _w = ((-1) ** _n / pi) * exp(-T) * poly(2 * T)
        else:
            factor1 = sqrt(factorial(m) / factorial(n))
            factor2 = ((-1) ** m / pi) * (sqrt(2) * (X + 1j * P)) ** (n - m) * exp(-T)
            _w = factor1 * factor2 * genlaguerre(m, n - m)(2 * T)
        return _w
    W = zeros([len(xRange), len(yRange)], dtype=complex)
    # Add all the contribution
    dim = prod(rho.shape[0])
    for m in range(dim):
        wmn = real(rho[m, m] * _wignerLaguerre(m, m))
        W += wmn
        for n in range(m + 1, dim):
            wmn = real(rho[m, n] * _wignerLaguerre(m, n))
            W += 2 * wmn
    return W


def coherent(dim: int = 2, alpha: complex = 0 + 0j):
    """
    Generate a coherent state using displacement operator.

    :param dim: dimension of the coherent state truncated.
    :param alpha: the eigenvalue of the annihilation operator.
    """

    # Initialize a vacuum state
    psi0 = basis(dim, 0)
    a = destroy(dim).matrix
    adag = dagger(a)

    # construct a displacement operator
    disp = linalg.expm(alpha * adag - conj(alpha) * a)
    psi = disp @ psi0

    return psi
