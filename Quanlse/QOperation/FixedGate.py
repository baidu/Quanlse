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
Fixed Gate Operation
"""
import math
from typing import TYPE_CHECKING, Optional

import numpy
import scipy.linalg

from Quanlse.QOperation import QOperation

if TYPE_CHECKING:
    from Quanlse.QRegPool import QRegStorage


class FixedGateOP(QOperation):
    """
    Fixed gates are set in built-in quantum tool chain.

    Only some solid gates with concrete definitions (without parameters) are set here,

    like Identity, Pauli X, Y, Z, Hadamard, Phase, T, CNOT (CX), etc.
    """

    def __init__(self, gate: str, bits: int, matrix: Optional[numpy.ndarray]) -> None:
        super().__init__(gate, bits, matrix)

    def __call__(self, *qRegList: 'QRegStorage', gateTime: Optional[float] = None) -> None:
        self._op(list(qRegList), gateTime)


X = FixedGateOP('X', 1,
                numpy.array([[0. + 0.j, 1. + 0.j],
                             [1. + 0.j, 0. + 0.j]])
                )
r"""
Pauli-X operator, also called NOT gate, means flipping the qubit.

For example:

:math:`X|0 \rangle = |1\rangle \quad \text{and} \quad  X|1 \rangle = |0\rangle` 

Matrix form:

:math:`X= \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}`
"""

Y = FixedGateOP('Y', 1,
                numpy.array([[0. + 0.j, 0. - 1.j],
                             [0. + 1.j, 0. + 0.j]])
                )
r"""
Pauli-Y operator, similar to Pauli-X operator

Matrix form:

:math:`Y= \begin{bmatrix}     0 & -i \\     i & 0    \end{bmatrix}`
"""

Z = FixedGateOP('Z', 1,
                numpy.array([[1. + 0.j, 0. + 0.j],
                             [0. + 0.j, -1. + 0.j]])
                )  # Pauli-Z operator
r"""
Pauli-Z operator, means changing a local phase

For example: 

:math:`Z(|0 \rangle +|1 \rangle )= |0 \rangle - |1 \rangle`  

Matrix form:

:math:`Z= \begin{bmatrix}     1 & 0 \\     0 & -1    \end{bmatrix}`
"""

H = FixedGateOP('H', 1,
                1 / numpy.sqrt(2) * numpy.array([[1, 1],
                                                 [1, -1]], dtype=complex)
                )
r"""
Hadamard gate: it's the most important single qubit gate. 

And it can prepare a superposed state via applied on zero state, i.e.,

:math:`H|0 \rangle =\frac{1}{\sqrt{2}}( |0 \rangle + |1 \rangle)`  

Matrix form:

:math:`H=\frac{1}{\sqrt{2}} \begin{bmatrix}     1 & 1 \\     1 & -1    \end{bmatrix}`
"""

S = FixedGateOP('S', 1,
                numpy.array([[1. + 0.j, 0. + 0.j],
                             [0. + 0.j, 0. + 1.j]])
                )
r"""
Phase gate or :math:`\frac{\pi}{4}`-gate, it equals :math:`S=e^{-i\frac{\pi}{4}}= Z^{\frac{1}{2}}`.

It changes a local phase, similar to Pauli-Z operator, i.e.,

:math:`S (|0 \rangle +|1 \rangle )= |0 \rangle +i  |1 \rangle`  

Matrix form:

:math:`S=  \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}`
"""

T = FixedGateOP('T', 1,
                numpy.array([[1. + 0.j, 0. + 0.j],
                             [0. + 0.j, numpy.exp(1j * numpy.pi / 4)]])
                )
r"""
T gate or :math:`(\frac{\pi}{8})`-gate, it equals :math:`T =e^{-i\frac{\pi}{8}}=Z^{\frac{1}{4}}`.

It changes a local phase, similar to :math:`Z` gate, i.e.,

:math:`T (|0 \rangle +|1 \rangle )= |0 \rangle + e^{i\frac{\pi}{4}} |1 \rangle`

Matrix form:

:math:`T = \begin{bmatrix} 1 & 0 \\ 0 & e^{i\frac{\pi}{4}} \end{bmatrix}`
"""

W = FixedGateOP('W', 1,
                (numpy.array([[0. + 0.j, 1. + 0.j],
                              [1. + 0.j, 0. + 0.j]])
                 +
                 numpy.array([[0. + 0.j, 0. - 1.j],
                              [0. + 1.j, 0. + 0.j]]))
                / math.sqrt(2)
                )
r"""
Matrix form:

:math:`W = X + Y = \begin{bmatrix} 0 & \frac{1-i}{\sqrt{2}} \\ \frac{1+i}{\sqrt{2}} & 0 \end{bmatrix}`
"""

SQRTW = FixedGateOP('SQRTW', 1,
                    scipy.linalg.sqrtm(
                        (numpy.array([[0. + 0.j, 1. + 0.j],
                                      [1. + 0.j, 0. + 0.j]])
                         +
                         numpy.array([[0. + 0.j, 0. - 1.j],
                                      [0. + 1.j, 0. + 0.j]]))
                        / math.sqrt(2))
                    )
r"""
Matrix form:

:math:`\sqrt{W} = \sqrt{X + Y} = \sqrt{\begin{bmatrix} 0 & \frac{1-i}{\sqrt{2}} \\ \frac{1+i}{\sqrt{2}} & 0 \end{bmatrix}}`
"""

CZ = FixedGateOP('CZ', 2,
                 numpy.array([
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, -1],
                 ], dtype=complex)
                 )
r"""
CZ gate, or control-Z gate, a native gate in the superconducting platform. It's similar to CNOT gate.

On the simulator mode (use inside the simulator): Matrix form:

:math:`CZ = \begin{bmatrix} 1 & 0  &0 & 0 \\ 0 & 1  & 0& 0 \\ 0 & 0  & 1& 0 \\ 0 & 0  & 0& -1 \end{bmatrix}`
"""

CR = FixedGateOP('CR', 2,
                 numpy.array([
                     [math.cos(-math.pi / 4), -(1j * math.sin(-math.pi / 4)), 0, 0],
                     [-(1j * math.sin(-math.pi / 4)), math.cos(-math.pi / 4), 0, 0],
                     [0, 0, math.cos(-math.pi / 4), (1j * math.sin(-math.pi / 4))],
                     [0, 0, (1j * math.sin(-math.pi / 4)), math.cos(-math.pi / 4)]
                 ], dtype=complex)
                 )
r"""
Cross-resonance gate, a native gate in the superconducting platform.

On the simulator mode (use inside the simulator): Matrix form:

:math:`CR = \begin{bmatrix} \cos(-\pi/4) & -i\sin(-\pi/4)  & 0 & 0 \\ -i\sin(-\pi/4) & \cos(-\pi/4)  & 0 & 0 \\ 
0 & 0  & \cos(-\pi/4) & i\sin(-\pi/4) \\ 0 & 0  & i\sin(-\pi/4) & \cos(-\pi/4) \end{bmatrix}`
"""

ISWAP = FixedGateOP('ISWAP', 2,
                    numpy.array([
                         [1, 0, 0, 0],
                         [0, 0, -1j, 0],
                         [0, -1j, 0, 0],
                         [0, 0, 0, 1],
                    ], dtype=complex)
                    )
r"""
ISWAP gate, a native gate in the superconducting platform.

On the simulator mode (use inside the simulator): Matrix form:

:math:`CZ = \begin{bmatrix} 1 & 0  &0 & 0 \\ 0 & 0  & -i & 0 \\ 0 & -i & 0 & 0 \\ 0 & 0  & 0 & 1 \end{bmatrix}`
"""

CNOT = FixedGateOP('CNOT', 2, numpy.array([
                         [1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0],
                    ], dtype=complex)
                    )
r"""
CNOT gate, or control-X gate.

On the simulator mode (use inside the simulator): Matrix form:

:math:`CNOT = \begin{bmatrix} 1 & 0  &0 & 0 \\ 0 & 1  & 0& 0 \\ 0 & 0  & 0 & 1 \\ 0 & 0  & 1 & 0 \end{bmatrix}`
"""

SWAP = FixedGateOP('SWAP', 2, numpy.array([                       
                            [1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]
                        ], dtype=complex)
                        )
r"""
SWAP gate.

On the simulator mode (use inside the simulator): Matrix form:

:math:`SWAP = \begin{bmatrix} 1 & 0  &0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}`
"""
