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
Functions used in ErrorMitigation module.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from itertools import combinations
from scipy import linalg
from typing import List

from Quanlse.QOperator import number, duff
from Quanlse.QHamiltonian import QHamiltonian
from Quanlse.QOperation import RotationGate, CircuitLine
from Quanlse.QPlatform.Error import ArgumentError
from Quanlse.Superconduct.SchedulerSupport import SchedulerSuperconduct
from Quanlse.Utils.Functions import project, tensor, expect, fromMatrixToAngles
from Quanlse.Utils.Plot import plotPulse
from Quanlse.Utils.Clifford import randomClifford
from Quanlse.Superconduct.SchedulerSupport.GeneratorRBPulse import SingleQubitCliffordPulseGenerator


def setupBasicHamiltonian(n: int = None) -> QHamiltonian:
    r"""
    Setup the system's Hamiltonian with prefixed parameters.

    :param n: number of qubits of the Hamiltonian
    :return: the basic Hamiltonian
    """
    if n is None:
        raise ArgumentError("Error in setupBasicHamiltonian(): number of qubits is not specified!")
    # Prefix the coupling strength of qubit pairs
    g = 0.0038 * 2 * pi
    # Anharmonicity
    anharm = -0.33 * 2 * pi
    # Qubit frequency list
    qubitFreq = [4.914 * 2 * pi,
                 5.114 * 2 * pi,
                 4.914 * 2 * pi,
                 5.114 * 2 * pi]
    # Set the rotating wave approximation frequency
    rwa = 4.914 * 2 * pi
    # Set the unit time for solving ODE
    dt = 1.0
    # Define the system Hamiltonian
    ham = QHamiltonian(subSysNum=n, sysLevel=3, dt=dt)
    # Add drift terms
    for q in range(n):
        ham.addDrift(number(3), q, qubitFreq[q] - rwa)
        ham.addDrift(duff(3), q, anharm)
    # Add coupling terms. If there is only one qubit, there is no coupling term.
    # We assume `full connectivity` of the quantum device,
    # thus there is a coupling term between arbitrary two qubits.
    for cq, tq in combinations(range(n), 2):
        ham.addCoupling([cq, tq], g)
    return ham


def fromCircuitToHamiltonian(circuit: List[CircuitLine] = None, useCliffordPulse: bool = True) -> QHamiltonian:
    r"""
    Convert a quantum circuit into the optimal Hamiltonian.

    :param circuit: the quantum circuit to be optimized. It is a list of `CircuitLine` objects, each `CircuitLine`
            specifies a quantum gate operating on several qubits
    :param useCliffordPulse: indicator for using the built-in Clifford pulse or not
    :return: The optimal Hamiltonian which is the optimal implementation of the unitary evolution of the gate sequence
    """
    # Step 1. Setup the basic Hamiltonian with prefixed parameters. These parameters are from the quantum devices.
    # Parse the number of qubits used in the quantum circuit
    n = numberOfQubitsOfCircuit(circuit)
    # Construct system Hamiltonian with basic information from quantum device
    ham = setupBasicHamiltonian(n)

    # Step 2. Use the Quanlse Scheduler to generate pulse sequences for the input quantum circuit
    if useCliffordPulse:
        scheduler = SchedulerSuperconduct(ham.dt, ham, generator=SingleQubitCliffordPulseGenerator(ham))
    else:
        scheduler = SchedulerSuperconduct(ham.dt, ham)
    # Translate the circuits to gates operated on the Scheduler
    for gate in circuit:
        if len(gate.qRegIndexList) == 1:
            gate.data(scheduler.Q[gate.qRegIndexList[0]])
        elif len(gate.qRegIndexList) == 2:
            gate.data(scheduler.Q[gate.qRegIndexList[0]], scheduler.Q[gate.qRegIndexList[1]])
        else:
            raise ArgumentError('Error in fromCircuitToHamiltonian(): unrecognized gate type!')

    # Run the scheduler
    scheduler.ham.job = scheduler.schedule()
    # Build the cache for the Hamiltonian
    scheduler.ham.buildCache()
    # Return the optimized Hamiltonian corresponding to the quantum circuit
    return scheduler.ham


def fromHamiltonianToOperator(ham: QHamiltonian) -> np.ndarray:
    r"""
    Call the `simulate` function to compute the corresponding evolution operator of the given Hamiltonian.

    :param ham: the Hamiltonian, with evolution time and wave sequences given
    :return: The evolution operator
    """
    result = ham.simulate(refreshCache=False)
    projectedEvolution = project(result.result[0]["unitary"], ham.subSysNum, ham.sysLevel, 2)
    return projectedEvolution


def numberOfQubitsOfCircuit(circuit: List[CircuitLine] = None) -> int:
    r"""
    Extract the number of qubits involved in the given quantum circuit.
    We implement this by collecting the qubit indices and count the number.

    :param circuit: the quantum circuit which is a list of `CircuitLine` objects.
    :return: number of qubits used in this quantum circuit
    """
    qubitIndices = []
    for gate in circuit:
        qubitIndices.extend(gate.qRegIndexList)
    qubitIndices = np.unique(qubitIndices)
    return len(qubitIndices)


def printHamiltonian(ham: QHamiltonian) -> None:
    r"""
    Print the basic information of a Hamiltonian object.
    This function is used within the ZNE module for illustrative purpose.

    :param ham: the given Hamiltonian
    :return: None
    """
    print('=================================================================')
    print('The basic information of the Hamiltonian is:')
    print(ham)
    print('Print the control pulses now ...')
    print("ham.job.endTimeDt = ", ham.job.endTimeDt)
    print(ham.ctrlCache)
    for key in ham.ctrlCache:
        for nowDt in range(0, ham.job.endTimeDt):
            print("ham.job.waveCache[{}][{}] = {}".format(key, nowDt, ham.job.waveCache[key][nowDt]))


def computeInverseGate(circuit: List[CircuitLine]) -> CircuitLine:
    r"""
    Compute the inverse gate of a sequence of single-qubit gates.
    Notice that a decomposition of single-qubit gates can be characterized via:

    :math:`U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)
    = e^{i(\alpha-\phi/2-\lambda/2)}
    \begin{bmatrix}
    \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\
    e^{i\phi}\sin(\theta/2) & e^{i(\phi+\lambda)}\cos(\theta/2)
    \end{bmatrix}
    = e^{i(\alpha-\phi/2-\lambda/2)} U'`

    We define the inverse to be the inverse of U'. That is,
    we do not consider the effect of global phases.
    Example:
        The input is `[X, Y, Z, H]`, the output will be `XYZH`

    :param circuit: the quantum circuit whose inverse to be computed.
            It is a list of `CircuitLine` objects, each `CircuitLine` specifies a quantum gate operating on several qubits.
    :return: the inverse gate in type of `CircuitLine`
    """
    if len(circuit) < 1:
        raise ArgumentError(f"The input gate sequence should contain at least one gate!")

    n = numberOfQubitsOfCircuit(circuit)
    product = np.identity(2 ** n)
    for item in circuit:
        product = item.data.getMatrix() @ product
    # Compute the inverse gate of the composite Clifford gates
    invMatrix = np.linalg.inv(product)
    # Transform the inverse gate to RotationGate.U format and add it to the circuit
    [_, theta, phi, lamda] = fromMatrixToAngles(invMatrix)
    return CircuitLine(RotationGate.U(theta, phi, lamda), [0])


def computeIdealEvolutionOperator(circuit: List[CircuitLine]) -> np.ndarray:
    """
    Compute the evolution unitary operator of the given circuit.

    :param circuit: the quantum circuit whose corresponding ideal evolution to be computed.
            It is a list of `CircuitLine` objects, each `CircuitLine` specifies a quantum gate operating on several qubits.
    :return: ndarray-type with shape (2^n, 2^n)
    """
    n = numberOfQubitsOfCircuit(circuit)
    U = tensor([np.identity(2)] * n)
    for gate in circuit:
        if len(gate.qRegIndexList) == 1:
            # Form a list of identities I \otimes ... \otimes I
            V = tensorWithSingleQubitGate(gate.data.getMatrix(), n, gate.qRegIndexList[0])
            U = V @ U
        elif len(gate.qRegIndexList) == 2:
            V = tensorWithTwoQubitGate(gate.data.getMatrix(), n, gate.qRegIndexList[0], gate.qRegIndexList[1])
            U = V @ U
        else:
            raise ArgumentError(
                'Error in computeIdealEvolutionOperator(): only single and two-qubit gates are supported!')
    return U


def computeIdealExpectationValue(rho: np.ndarray,
                                 circuit: List[CircuitLine] = None,
                                 A: np.ndarray = None) -> float:
    r"""
    Compute the ideal expectation value of the following form

    :math:`{\rm{Tr}}[A C_n\cdots C_1 \rho C_1^\dagger \cdots C_n^\dagger]`

    where each :math:`C_i` is a quantum gate.

    :param rho: the input quantum state
    :param circuit: the quantum circuit. It is a list of `CircuitLine` objects,
            each `CircuitLine` specifies a quantum gate operating on several qubits.
    :param A: the quantum observable
    :return: the ideal expectation value
    """
    n = numberOfQubitsOfCircuit(circuit)
    if (rho.shape[0] != 2 ** n) or (rho.shape[1] != 2 ** n):
        raise ArgumentError("the input state must have the same dimension of the quantum circuit!")
    if (A.shape[0] != 2 ** n) or (A.shape[1] != 2 ** n):
        raise ArgumentError("the observable must have the same dimension of the quantum circuit!")

    U = computeIdealEvolutionOperator(circuit)
    return expect(A, U @ rho @ U.conj().T)


def tensorWithSingleQubitGate(singleQubitOperator: np.ndarray, n: int, qubitIndex: int) -> np.ndarray:
    """
    Given a single-qubit gate, we compute its tensor unitary matrix expanded
    to the whole Hilbert space (totally n qubits).

    :param singleQubitOperator: ndarray with shape (2,2)
    :param n: number of qubits
    :param qubitIndex: index of the qubit with the subsequent single-qubit operation
    :return: ndarray with shape (2^n, 2^n)
    """
    if qubitIndex not in range(n):
        raise ArgumentError("The qubitIndex is out of range!")
    qubitGateList = [np.identity(2)] * n
    # Substitute the i-th gate with the given gate
    qubitGateList[qubitIndex] = singleQubitOperator
    # Use tensor function to obtain the composite unitary operator
    U = tensor(qubitGateList)
    return U


def tensorWithTwoQubitGate(twoQubitOperator: np.ndarray, n: int, cq: int, tq: int) -> np.ndarray:
    """
    Much similar to the `tensorWithSingleQubitGate` function,
    this function aims to compute the tensor unitary matrix expanded
    to the whole Hilbert space (totally n qubits) of a two-qubit gate.

    :param twoQubitOperator: ndarray with shape (4,4)
    :param n: number of qubits
    :param cq: control qubit index
    :param tq: target qubit index
    :return: ndarray with shape (2^n, 2^n)
    """
    if cq not in range(n) or tq not in range(n):
        raise ArgumentError("The control or the target qubit is out of range!")

    qubitGateList = [np.identity(2)] * (n - 1)
    qubitGateList[0] = twoQubitOperator
    U = tensor(qubitGateList).reshape([2] * 2 * n)
    idx = np.repeat(-1, n)
    idx[cq] = 0
    idx[tq] = 1
    idx[idx < 0] = range(2, n)
    idx = idx.tolist()
    idxLatter = [i + n for i in idx]
    U = np.transpose(U, idx + idxLatter).reshape([2 ** n, 2 ** n])
    return U


def plotHamiltonianPulse(ham: QHamiltonian) -> None:
    r"""
    Print the pulse of a Hamiltonian object.

    :param ham: the given Hamiltonian
    :return: None
    """
    if ham.ctrlCache is None or len(ham.ctrlCache) == 0:
        raise ArgumentError("in plotHamiltonianPulse(): the control cache is not set!")
    # Create two empty lists
    x = []
    y = []
    yLabel = []
    colors = []
    for key in ham.ctrlCache:
        aList = np.array(ham.waveCache[key])
        tList = np.linspace(0, len(aList) * ham.dt, num=len(aList))
        y.append(list(aList))
        x.append(list(tList))
        yLabel.append('Amp (a.u.)')
        colors.append('blue')
    plotPulse(x, y, xLabel=f'Time (ns)', yLabel=yLabel, title=list(ham.ctrlCache), color=colors, dark=False)
    plt.show()


def globalPhase(U: np.ndarray) -> float:
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
    alpha = - np.angle(coe)
    return alpha


def removeGlobalPhase(U: np.ndarray) -> np.ndarray:
    r"""
    Remove the global phase of a 2*2 unitary matrix.
    Each 2*2 unitary matrix can be equivalently characterized as:

    :math:`U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)`

    We aim to remove the global phase `e^{i\alpha}` from the unitary matrix.
    See Theorem 4.1 in `Nielsen & Chuang`'s book for details.

    :param U: the matrix representation of the 2*2 unitary operator
    :return: the unitary matrix whose global phase has been removed
    """
    alpha = globalPhase(U)
    U = U * np.exp(- 1j * alpha)
    return U


def rotationXZCircuit(numSeq: int = 5) -> List[CircuitLine]:
    r"""
    Return a list including serial X-Z rotation gates, while the whole effect of all elements is equal to a X gate.

    :math:`U_j = R_z(\frac{4j\pi}{n})R_x(\frac{j\pi}{n})R_x(\frac{-(j-1)\pi}{n})R_z(\frac{-4(j-1)\pi}{n})`

    :param numSeq: length of the X-Z rotation gates sequences
    :return: a list including serial X-Z rotation gates ("numSeq" elements in total)
    """
    circuit = []
    n = numSeq
    for k in range(1, numSeq + 1):
        rzp = RotationGate.RZ(4 * k * pi / n).getMatrix()
        rxp = RotationGate.RX(k * pi / n).getMatrix()
        rxm = RotationGate.RX(-(k - 1) * pi / n).getMatrix()
        rzm = RotationGate.RZ(-4 * (k - 1) * pi / n).getMatrix()
        Uk = rzp @ rxp @ rxm @ rzm
        _, theta, phi, lamda = fromMatrixToAngles(Uk)
        circuit.append(CircuitLine(RotationGate.U(theta, phi, lamda), [0]))
    return circuit


def randomCircuit(qubits: int = 1, numSeq: int = 5, seed: int = None, type: str = 'clifford') -> List[CircuitLine]:
    r"""
    Randomly generate a numSeq-length quantum circuit based on `size` number of Clifford operators,
    whose number of qubits is given by `qubits`. Currently only single-qubit circuit is supported.

    :param qubits: number of qubits
    :param numSeq: number of random Clifford gates
    :param seed: random number seed
    :param type: 'clifford' or other optional types
    :return: a circuit (`list` type) including a series of gates (`CircuitLine` type)
    """
    if type != 'clifford':
        raise ArgumentError("This function currently can only generate random gates of Clifford type!")
    cliffordGates = randomClifford(qubits, numSeq, seed=seed)
    circuit = [CircuitLine(gate, [0]) for gate in cliffordGates]

    return circuit
