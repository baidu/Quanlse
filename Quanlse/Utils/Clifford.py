#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
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
This file aims to collect functions related to the Clifford group.
"""

from typing import List
import numpy as np

from Quanlse.QOperation.FixedGate import FixedGateOP
from Quanlse.QOperation.RotationGate import RX, RY, RZ
from Quanlse.QPlatform.Error import ArgumentError

# The basic elements of Clifford gates
x90p = RX(np.pi / 2).getMatrix()  # Rx+
x90m = RX(-np.pi / 2).getMatrix()  # Rx-
xp = RX(np.pi).getMatrix()  # Rx
y90p = RY(np.pi / 2).getMatrix()  # Ry+
y90m = RY(-np.pi / 2).getMatrix()  # Ry-
yp = RY(np.pi).getMatrix()  # Ry
z90p = RZ(np.pi / 2).getMatrix()  # Rz+
z90m = RZ(-np.pi / 2).getMatrix()  # Rz-
zp = RZ(np.pi).getMatrix()  # Rz
Id = np.array([[1, 0], [0, 1]], dtype=np.complex128)  # Identity

# Generate single-qubit Clifford matrices
C1 = FixedGateOP('C1', 1, Id)
C2 = FixedGateOP('C2', 1, xp)
C3 = FixedGateOP('C3', 1, yp)
C4 = FixedGateOP('C4', 1, xp @ yp)
C5 = FixedGateOP('C5', 1, x90p @ y90p)
C6 = FixedGateOP('C6', 1, x90p @ y90m)
C7 = FixedGateOP('C7', 1, x90m @ y90p)
C8 = FixedGateOP('C8', 1, x90m @ y90m)
C9 = FixedGateOP('C9', 1, y90p @ x90p)
C10 = FixedGateOP('C10', 1, y90p @ x90m)
C11 = FixedGateOP('C11', 1, y90m @ x90p)
C12 = FixedGateOP('C12', 1, y90m @ x90m)
C13 = FixedGateOP('C13', 1, x90p)
C14 = FixedGateOP('C14', 1, x90m)
C15 = FixedGateOP('C15', 1, y90p)
C16 = FixedGateOP('C16', 1, y90m)
C17 = FixedGateOP('C17', 1, x90m @ y90p @ x90p)
C18 = FixedGateOP('C18', 1, x90m @ y90m @ x90p)
C19 = FixedGateOP('C19', 1, xp @ y90p)
C20 = FixedGateOP('C20', 1, xp @ y90m)
C21 = FixedGateOP('C21', 1, yp @ x90p)
C22 = FixedGateOP('C22', 1, yp @ x90m)
C23 = FixedGateOP('C23', 1, x90p @ y90p @ x90p)
C24 = FixedGateOP('C24', 1, x90m @ y90p @ x90m)

# Set the single-qubit Clifford gate set
clifford1q = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10,
              C11, C12, C13, C14, C15, C16, C17, C18, C19, C20,
              C21, C22, C23, C24]


def randomClifford(qubits: int = 1, size: int = 1, seed: int = None) -> List[FixedGateOP]:
    r"""
    Randomly generate `size` number amount of Clifford operators,
    whose number of qubit(s) is given by `qubits`.

    :param qubits: the number of qubits of each Clifford operator
    :param size: the number of Clifford operators generated
    :param seed: the seed used for randomly generating Clifford operators
    :return: A list of generated Clifford operators in U3 representation.
            Note that the return type is always a list.
    """
    if qubits != 1:
        raise ArgumentError(f'in random_Clifford(): only support single-qubit Clifford now!')

    if seed is None:
        idx = np.random.randint(0, 24, size)
    else:
        np.random.seed(seed)
        idx = np.random.randint(0, 24, size)

    cliffordRandomList = [clifford1q[idx[i]] for i in range(size)]

    return cliffordRandomList
