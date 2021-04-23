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

from math import sqrt, factorial, exp
import numpy


def dagger(matrix: numpy.ndarray) -> numpy.ndarray:
    """
    Return the conjugate transpose of a given matrix.

    :param matrix: the given matrix
    :return: the conjugate transposed matrix
    """
    return numpy.array(numpy.conjugate(numpy.transpose(matrix)), order="C")


def destroy(d: int = 2) -> numpy.ndarray:
    r"""
    Return the matrix form of a annihilation operator :math:`a`
    with a dimension of :math:`d` (:math:`d` is an integer):

    :math:`a = \begin{bmatrix} 0 & \sqrt{1} & 0 & \cdots & 0 \\ 0 & 0 & \sqrt{2} & \cdots & 0 \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \cdots & \sqrt{d} \\
    0 & 0 & 0 & \cdots & 0
    \end{bmatrix}`

    :param d: the dimension of the annihilation operator
    :return: matrix form of the annihilation operator
    """
    mat = numpy.zeros((d, d), dtype=complex, order="C")
    for i in range(0, d - 1):
        mat[i, i + 1] = sqrt(i + 1) + 0 * 1j
    return mat


def create(d: int = 2) -> numpy.ndarray:
    r"""
    Return the matrix form of a creation operator :math:`a^{\dagger}`
    with a dimension of :math:`d` (:math:`d` is an integer):

    :math:`a^{\dagger} = \begin{bmatrix} 0 & 0 & \cdots & 0 & 0 \\ \sqrt{1} & 0 & \cdots & 0 & 0 \\
    0 & \sqrt{2} & \cdots & 0 & 0 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\
    0 & 0 & \cdots & \sqrt{d} & 0
    \end{bmatrix}`

    :param d: the dimension of the creation operator
    :return: matrix form of the creation operator
    """
    return dagger(destroy(d))


def driveX(d: int = 2) -> numpy.ndarray:
    r"""
    Return the matrix of a X-channel driven operator: :math:`(a + a^{\dagger}) / 2`.

    :param d: the dimension of the X-channel driven operator
    :return: matrix of form the X-channel driven operator
    """
    return (destroy(d) + create(d)) / 2


def driveY(d: int = 2) -> numpy.ndarray:
    r"""
    Return the matrix of a Y-channel driven operator: :math:`i (a - a^{\dagger}) / 2`.

    :param d: the dimension of the Y-channel driven operator
    :return: matrix form of the Y-channel driven operator
    """
    return 1j * (destroy(d) - create(d)) / 2


def sigmaX() -> numpy.ndarray:
    r"""
    Return the matrix form of a Pauli-X operator:
    :math:`\sigma_x = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}`.

    :return: matrix form of the Pauli-X operator
    """
    return numpy.array([[0, 1], [1, 0]], dtype=complex, order="C")


def sigmaY() -> numpy.ndarray:
    r"""
    Return the matrix form of a Pauli-Y operator:
    :math:`\sigma_y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}`.

    :return: matrix form of the Pauli-Y operator
    """
    return numpy.array([[0, -1j], [1j, 0]], dtype=complex, order="C")


def sigmaZ() -> numpy.ndarray:
    r"""
    Return the matrix form of a Pauli-Z operator:
    :math:`\sigma_0 = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}`.

    :return: matrix form of the Pauli-Z operator
    """
    return numpy.array([[1, 0], [0, -1]], dtype=complex, order="C")


def sigmaI() -> numpy.ndarray:
    r"""
    Return the matrix form of an identity operator:
    :math:`\sigma_y = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`.

    :return: an identity matrix
    """
    return numpy.array([[1, 0], [0, 1]], dtype=complex, order="C")


def number(d: int = 2) -> numpy.ndarray:
    r"""
    Return the matrix form of the number operator:
    :math:`a^{\dagger}a`.

    :param d: the dimension of the number operator
    :return: matrix form of the number operator
    """
    return numpy.dot(create(d), destroy(d))


def duff(d: int = 2) -> numpy.ndarray:
    r"""
    Return the matrix form of the Duffing operator:
    :math:`a^{\dagger}a^{\dagger}aa`.

    :param d: the dimension of the Duffing operator
    :return: matrix form of the Duffing operator
    """
    return numpy.dot(numpy.dot(create(d), create(d)), numpy.dot(destroy(d), destroy(d)))


def basis(d: int, state: int) -> numpy.ndarray:
    r"""
    Return the vector of state operator:
    :math:`|0\rangle, |1\rangle, \cdots`.

    :param d: the dimension of the Hilbert space
    :param state: the index of the state
    :return: matrix form of the state operator
    """
    assert d > state, "The input state index exceeds system dimension"
    matrix = numpy.zeros((d, 1), dtype=complex)
    matrix[state] = 1.0
    return matrix


def projector(a: numpy.ndarray, b: numpy.ndarray = None) -> numpy.ndarray:
    r"""
    Return the matrix form of a state: :math:`|a\rangle \langle b|`.

    :param a: ket operator
    :param b: bra operator
    :return: the outer product of the two operators :math:`|a\rangle` and :math:`(|b\rangle)^{\dagger}`
    """
    if b is None:
        returnMatrix = numpy.dot(a, dagger(a))
    else:
        returnMatrix = numpy.dot(a, dagger(b))
    return returnMatrix


def coherent(alpha: float, n: int) -> numpy.ndarray:
    r"""
    Return the state vector of the coherent state:
    :math:`e^{-\frac{1}{2}|\alpha|^2}\sum_{i=0}^{n-1}\frac{\alpha^i}{\sqrt{n!}}|i\rangle`

    :param alpha: the eigenstate of the annihilation operator.
    :param n: the highest level truncated.

    :return: the state vector of the coherent state.
    """
    return exp(-0.5 * pow(alpha, 2)) * sum([pow(alpha, i) / sqrt(factorial(i)) * basis(n, i) for i in range(n)])
