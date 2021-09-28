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
Quanlse QOperator class.
"""

import pickle
import base64
from math import sqrt
from numpy import ndarray, array, zeros, conjugate, transpose, dot
from typing import Union

from Quanlse.QPlatform import Error


class QOperator:
    """
    In Quanlse, operators are stored in QOperator objects, which keep track of the
    operator's purpose, matrix form, coefficients and the qubits it corresponds to.

    :param name: a user-given name
    :param matrix: corresponding matrix of the operator
    :param onSubSys: subsystem number
    :param coef: corresponding coefficient
    """
    def __init__(self, name: str, matrix: ndarray = None, onSubSys: int = None,
                 coef: Union[float, complex] = 1.0) -> None:
        """
        The constructor of the QOperator class.
        """
        self.name = name  # type: str
        self.matrix = matrix  # type: ndarray
        self.onSubSys = onSubSys  # type: int
        self.coef = coef  # type: float

    @property
    def matrix(self):
        """
        Return the corresponding matrix (type: ndarray)
        """
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: ndarray = None):
        """
        Setter function for the matrix

        :param matrix: the matrix to be set to
        """
        if matrix is not None and not isinstance(matrix, ndarray):
            raise Error.ArgumentError("matrix must be a numpy.ndarray!")
        self._matrix = matrix

    @property
    def name(self):
        """
        Return the corresponding name of the operator.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """
        Modify the name of the operator.

        :param name: name to be changed to
        """
        if not isinstance(name, str):
            raise Error.ArgumentError("name must be a str!")
        self._name = name

    @property
    def onSubSys(self):
        """
        Return the corresponding subsystem number of the operator.
        """
        return self._onSubSys

    @onSubSys.setter
    def onSubSys(self, onSubSys: int):
        """
        Modify the corresponding subsystem of the operator.

        :param onSubSys: subsystem to be changed to
        """
        if onSubSys is None:
            self._onSubSys = None
        else:
            if not isinstance(onSubSys, int):
                raise Error.ArgumentError(f"onSubSys must be an integer, instead of {type(onSubSys)}!")
            self._onSubSys = onSubSys

    @property
    def coef(self):
        """
        Coefficient of the operator
        """
        return self._coef

    @coef.setter
    def coef(self, coef: Union[float, complex] = 1.0):
        """
        Setter for the coefficient

        :param: coefficient to be set to
        """
        if coef is None:
            self._coef = 1.0
        else:
            if not (isinstance(coef, float) or isinstance(coef, complex)):
                raise Error.ArgumentError(f"coef must be a float or complex, instead of {type(coef)}!")
            self._coef = coef

    def dump(self) -> str:
        """
        Return base64 encoded string.

        :return: a base64 encoded string
        """
        # Dump the object
        byteStr = pickle.dumps(self)
        base64str = base64.b64encode(byteStr)
        return base64str.decode()

    @staticmethod
    def load(base64Str: str) -> 'QOperator':
        """
        Create object from base64 encoded string.

        :return: a QOperator object
        """
        byteStr = base64.b64decode(base64Str.encode())
        obj = pickle.loads(byteStr)  # type: QOperator
        return obj


def sigmaI(arg=None) -> QOperator:
    r"""
    Matrix form of the Pauli-I operator:

    :math:`\hat{\sigma}_I = \begin{bmatrix} 1 & 0 \\ 0 & 1
    \end{bmatrix}`

    :return: a QOperator object
    """
    qo = QOperator("sigmaI", matrix=array([
        [1, 0],
        [0, 1]
    ], dtype=complex))
    return qo


def sigmaX(arg=None) -> QOperator:
    r"""
    Matrix form of the Pauli-X operator:

    :math:`\hat{\sigma}_X = \begin{bmatrix} 0 & 1 \\ 1 & 0
    \end{bmatrix}`

    :return: a QOperator object
    """
    qo = QOperator("sigmaX", matrix=array([
        [0, 1],
        [1, 0]
    ], dtype=complex))
    return qo


def sigmaY(arg=None) -> QOperator:
    r"""
    Matrix form of the Pauli-Y operator:

    :math:`\hat{\sigma}_Y = \begin{bmatrix} 0 & -i \\ i & 0
    \end{bmatrix}`

    :return: a QOperator object
    """
    qo = QOperator("sigmaY", matrix=array([
        [0, -1j],
        [1j, 0]
    ], dtype=complex))
    return qo


def sigmaZ(_=None) -> QOperator:
    r"""
    Matrix form of the Pauli-Z operator:

    :math:`\hat{\sigma}_Z = \begin{bmatrix} 1 & 0 \\ 0 & -1
    \end{bmatrix}`

    :return: a QOperator object
    """
    qo = QOperator("sigmaZ", matrix=array([
        [1, 0],
        [0, -1]
    ], dtype=complex))
    return qo


def number(d: int = 2) -> QOperator:
    r"""
    Return the matrix form of the number operator:
    :math:`\hat{a}^{\dagger}\hat{a}`.

    :param d: the dimension of the number operator
    :return: matrix form of the number operator
    """
    return QOperator("number", dot(adagger(d), a(d)))


def uWave(d: int = 2) -> QOperator:
    r"""
    Return the matrix of a microwave driven operator: :math:`(\hat{a} + \hat{a}^{\dagger})`.

    :param d: the dimension of the microwave driven operator
    :return: matrix of form the microwave driven operator
    """
    return QOperator("uWave", a(d) + adagger(d))


def flux(d: int = 2) -> QOperator:
    r"""
    Return the matrix of a magnetic flux driven operator: :math:`\hat{a}^{\dagger}\hat{a}`.

    :param d: the dimension of the magnetic flux channel driven operator
    :return: matrix of form the magnetic flux channel driven operator
    """
    return QOperator("flux", dot(adagger(d), a(d)))


def driveX(d: int = 2) -> QOperator:
    r"""
    Return the matrix of a X-channel driven operator: :math:`(\hat{a} + \hat{a}^{\dagger}) / 2`.

    :param d: the dimension of the X-channel driven operator
    :return: matrix of form the X-channel driven operator
    """
    return QOperator("driveX", (a(d) + adagger(d)) / 2)


def driveY(d: int = 2) -> QOperator:
    r"""
    Return the matrix of a Y-channel driven operator: :math:`i (\hat{a} - \hat{a}^{\dagger}) / 2`.

    :param d: the dimension of the Y-channel driven operator
    :return: matrix form of the Y-channel driven operator
    """
    return QOperator("driveY", 1j * (a(d) - adagger(d)) / 2)


def driveZ(d: int = 2) -> QOperator:
    r"""
    Return the matrix of a Z-channel driven operator: :math:`\hat{a}^{\dagger}\hat{a}`.

    :param d: the dimension of the Y-channel driven operator
    :return: matrix form of the Y-channel driven operator
    """
    return QOperator("driveZ", dot(adagger(d), a(d)))


def duff(d: int = 2) -> QOperator:
    r"""
    Return the matrix form of the Duffing operator:
    :math:`\hat{a}^{\dagger}\hat{a}^{\dagger}\hat{a}\hat{a}`.

    :param d: the dimension of the Duffing operator
    :return: matrix form of the Duffing operator
    """
    mat = dot(dot(adagger(d), adagger(d)), dot(a(d), a(d)))
    return QOperator("duff", mat)


def destroy(d: int = 2) -> QOperator:
    r"""
    Return the matrix form of an annihilation operator :math:`\hat{a}`
    with a dimension of :math:`d` (:math:`d` is an integer):

    :math:`\hat{a} = \begin{bmatrix} 0 & \sqrt{1} & 0 & \cdots & 0 \\ 0 & 0 & \sqrt{2} & \cdots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & \sqrt{d-1} \\
    0 & 0 & 0 & \cdots & 0
    \end{bmatrix}`

    :param d: the dimension of the annihilation operator
    :return: matrix form of the annihilation operator
    """
    return QOperator("a", a(d))


def create(d: int = 2) -> QOperator:
    r"""
    Return the matrix form of a creation operator :math:`\hat{a}^{\dagger}`
    with a dimension of :math:`d` (:math:`d` is an integer):

    :math:`\hat{a}^{\dagger} = \begin{bmatrix} 0 & 0 & \cdots & 0 & 0 \\ \sqrt{1} & 0 & \cdots & 0 & 0 \\
    0 & \sqrt{2} & \cdots & 0 & 0 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\
    0 & 0 & \cdots & \sqrt{d-1} & 0
    \end{bmatrix}`

    :param d: the dimension of the creation operator
    :return: matrix form of the creation operator
    """
    return QOperator("adag", adagger(d))


def dagger(matrix: ndarray) -> ndarray:
    """
    Return the conjugate transpose of a given matrix.

    :param matrix: the given matrix
    :return: the conjugate transposed matrix
    """
    return array(conjugate(transpose(matrix)), order="C")


def a(d: int = 2) -> ndarray:
    """
    Return the numpy matrix of destroy operator.

    :return: a numpy matrix
    """
    mat = zeros((d, d), dtype=complex, order="C")
    for i in range(0, d - 1):
        mat[i, i + 1] = sqrt(i + 1) + 0 * 1j
    return mat


def adagger(d: int = 2) -> ndarray:
    """
    Return the numpy matrix of creation operator.

    :return: a numpy matrix
    """
    return dagger(a(d))
