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
Hamiltonian
"""

import copy
import json
from math import floor
from typing import Dict, Any, List, Union, Tuple, Callable

import matplotlib.pyplot as plt
from scipy.linalg import expm
import numpy
import time

from Quanlse.QPlatform.Utilities import numpyMatrixToDictMatrix, dictMatrixToNumpyMatrix
from Quanlse.Utils.Waveforms import play, makeWaveData
from Quanlse.Utils import Operator
from Quanlse.Utils.Plot import plotPulse


def createHam(title: str, dt: float, qubitNum: int, sysLevel: int) -> Dict[str, Any]:
    """
    Create a new empty Hamiltonian dictionary with specified system level and qubit number.

    :param title: the name of this Hamiltonian dictionary
    :param dt: arbitrary wave generator's (AWG) sampling time interval (also stands for the step size for
        the piecewise-constant quantum simulation algorithm)
    :param qubitNum: the number of qubits
    :param sysLevel: the dimension of the Hilbert space of a single qubit
    :return: the Hamiltonian dictionary
    """
    ham = {"file": {
        "title": title
    }, "circuit": {
        "dt": dt,
        "qubits": qubitNum,
        "sys_level": sysLevel,
        "max_time_dt": 0,
        "max_time_ns": 0
    }, "drift": {}, "control": {}}

    assert qubitNum >= 1, "Qubit number should be greater than 0."
    assert sysLevel >= 2, "System should be greater than 1."

    # Initialize cache
    clearCache(ham)

    return ham


def addDrift(ham: Dict[str, Any], name: str, onQubits: Union[int, List[int]] = None,
             matrices: Union[numpy.ndarray, List[numpy.ndarray]] = None, amp: float = 1) -> None:
    r"""
    Add a time-independent drift term to the Hamiltonian.

    :param ham: the Hamiltonian dictionary
    :param name: the name of this term
    :param onQubits: the index list of the qubit(s) which the drift term(s) is acting upon
    :param matrices: the matrices corresponding to the drift term(s) acting on the qubit(s) listed in onQubits
    :param amp: the prefactor of this term
    :return: None

    If ``onQubits`` is None, parameter matrices should pass a matrix with the same dimension as the system.

    **Example 1** (add drift terms):

    Given a three-qubit system, the users could run the following code:

    .. code-block:: python

            qham.addDrift(ham, "Operator1", onQubits=[0, 2], matrices=[number(2), number(2)], amp=omega1)

    The code above allows the users to efficiently add the specfied drift operator to the Hamiltonian:

    :math:`H = \omega_1 (a^{\dagger}a) \otimes I \otimes (a^{\dagger}a),`

    where :math:`a,a^{\dagger}` are the annihilation and creation operators; :math:`I` is the identity operator.

    In Quanlse, we provide a function for automatically building the global operator. For example,
    before you run the manual simulation algorithm, please use the function ``Hamiltonian.buildOperatorCache()``
    to take the tensor product of the operators provided in the ``matrices`` (according to the qubit indices
    ``onQubits``) to span the Hilbert space of three qubits.

    When we run ``addDrift()`` more than once, linear combinations of multiple operators will be
    added to the Hamiltonian.

    Finally, the total drift term is given:

    :math:`H_{\rm drift} = \sum_{k=1}^N H_k.`
    """

    sysLevel = ham["circuit"]["sys_level"]
    qubitNum = ham["circuit"]["qubits"]

    if onQubits is None:
        # Input the complete matrices directly.
        dim = sysLevel ** qubitNum
        assert numpy.shape(matrices) == (dim, dim), "Dimension does not match."
        matrices = [matrices]
    else:
        # Input the operators on certain qubits.
        if isinstance(onQubits, int):
            assert onQubits < qubitNum, "Qubit index is out of range."
            onQubits = [onQubits]
            if isinstance(matrices, numpy.ndarray):
                matrices = [matrices]
        elif isinstance(onQubits, list):
            maxQubitId = max(onQubits)
            assert maxQubitId < qubitNum, "Qubit index is out of range."
        else:
            assert False, "Variable onQubits should be a list of int."

    # Check whether the name exists
    assert name not in (ham["drift"]).keys(), f"Drift term name ({name}) already existed."

    term = {
        "on_qubits": onQubits,
        "matrices": matrices,
        "amp": amp
    }

    ham["drift"][name] = term


def addCoupling(ham: Dict[str, Any], name: str, onQubits: Union[int, List[int]], g: float = 1.0) -> None:
    r"""
    Add a coupling term to the Hamiltonian.

    :param ham: the Hamiltonian dictionary
    :param name: the name of this term
    :param onQubits: the index list of the terms working on
    :param g: coupling strength
    :return: None.

    Two qubits' indexes should be provided; and the coupling term is added between them:

    :math:`H_{\rm coupling} = g (a_{\rm q1}^{\dagger} a_{\rm q2} + a_{\rm q1} a_{\rm q2}^{\dagger}),`

    where :math:`a_q,a_q^{\dagger}` are the annihilation and creation operators on the :math:`q`-th qubit.

    **Example 1** (add coupling terms):

    Given a three-qubit system, the users could run following codes:

    .. code-block:: python

            qham.addCoupling(ham, "Operator1", onQubits=[0, 2], g=0.0227 * 2 * pi)

    The code above allows the users to efficiently add the specfied drift operator to the Hamiltonian:

    :math:`H = 0.14262830631 (a_0 \otimes I \otimes a_2^{\dagger} + a_0^{\dagger} \otimes I \otimes a_2).`
    """
    assert len(onQubits) == 2, "Coupling term should be defined on two qubits."
    # Obtain the system energy level.
    d = ham["circuit"]["sys_level"]
    # Construct matrices.
    matrices = [Operator.destroy(d), Operator.create(d)]
    matricesHc = [Operator.create(d), Operator.destroy(d)]
    # Add drift terms
    addDrift(ham, f"{name}", onQubits=onQubits, matrices=matrices, amp=g)
    addDrift(ham, f"{name}(hc)", onQubits=onQubits, matrices=matricesHc, amp=g)


def addControl(ham: Dict[str, Any], name: str, onQubits: Union[int, List[int]] = None,
               matrices: Union[numpy.ndarray, List[numpy.ndarray]] = None) -> None:
    r"""

    Add a control term to the Hamiltonian.

    :param ham: the Hamiltonian dictionary
    :param name: the name of this term
    :param onQubits: the list of index(es) of the qubit(s) which the control term(s) are acting upon
    :param matrices: the matrices corresponding to the control term(s) acting on the qubit(s) listed in onQubits
    :return: None

    If ``onQubits`` is None, parameter matrices should pass a matrix with the same dimension as the system.

    **Example 1** (add control terms):

    Given a three-qubit system, the users could run following codes:

    .. code-block:: python

            qham.addControl(ham, "Operator1", onQubits=[0, 2], matrices=[someOp1, someOp2])

    the codes above will add the specified control operator to the Hamiltonian:

    :math:`H_{\rm 1} = {\rm Op}_1 \otimes I \otimes {\rm Op}_2,`

    where :math:`a,a^{\dagger}` are the annihilation and creation operators; :math:`I` is the identity matrix.

    In Quanlse we provide a function for automatically building the global operator. For example,
    before you run the manual simulation algorithm, please use the function ``Hamiltonian.buildOperatorCache()``
    to take the tensor product of the operators provided in the ``matrices`` (according to the qubit indices
    ``onQubits``) to span the Hilbert space of three qubits.

    When we run ``addControl()`` more than once, linear combinations of multiple operators will be
    added to the Hamiltonian.

    Finally, the total control term is given:

    :math:`H_{\rm control} = \sum_{k=1}^N H_k.`

    **Note**: waveforms should be added to the control terms, using the functions
    ``Hamiltonian.addWave()`` or ``Hamiltonian.setWave()``.
    """

    sysLevel = ham["circuit"]["sys_level"]
    qubitNum = ham["circuit"]["qubits"]

    if onQubits is None:
        # Input the complete matrices directly.
        dim = sysLevel ** qubitNum
        assert numpy.shape(matrices) == (dim, dim), "Dimension does not match."
        matrices = [matrices]
    else:
        # Add the matrix on exact qubit
        if isinstance(onQubits, int):
            assert onQubits < qubitNum, "Qubit index is out of range."
            onQubits = [onQubits]
            if isinstance(matrices, numpy.ndarray):
                matrices = [matrices]
        elif isinstance(onQubits, list):
            maxQubitId = max(onQubits)
            assert maxQubitId < qubitNum, "Qubit index is out of range."
        else:
            assert False, "Variable onQubits should be a list of int."

    # Check whether the name exists
    assert name not in (ham["control"]).keys(), f"Control term name ({name}) already existed."

    term = {
        "on_qubits": onQubits,
        "matrices": matrices,
        "waveforms": []
    }

    ham["control"][name] = term


def subSystem(ham: Dict[str, Any], onQubits: Union[int, List[int]], title: str = "") -> Dict[str, Any]:
    """
    Extract a specified subsystem from a given Hamiltonian.
    The drift and control terms local to the specified subsystem remains.

    :param ham: the Hamiltonian dictionary
    :param onQubits: the qubits constitute the subsystem
    :param title: the title of the subsystem
    :return: the Hamiltonian dictionary of the subsystem
    """
    subHam = copy.deepcopy(ham)

    # clear cache
    clearCache(subHam)

    # Set title
    if title == "":
        subHam["file"]["title"] = f"{ham['file']['title']} (extracted)"
    else:
        subHam["file"]["title"] = title

    subHam["control"] = {}
    subHam["drift"] = {}

    # Information about the sub system
    if isinstance(onQubits, int):
        subQubits = 1
        indexMapping = {onQubits: 0}
    elif isinstance(onQubits, list):
        subQubits = len(onQubits)
        # qubit index mapping
        indexMapping = {}
        for qid, tar in enumerate(onQubits):
            indexMapping[tar] = qid
    else:
        assert False, "Unsupported input of qubitNum, it should be an int or a list."

    def mapping(index: Union[int, List[int]]) -> Union[int, List[int]]:
        """ Map the `onQubits` index from original Ham to the extracted Ham """
        if isinstance(index, int):
            return indexMapping[index]
        else:
            mappedList = []
            for item in index:
                mappedList.append(indexMapping[item])
            return mappedList

    def allIn(listA: Union[int, List[int]], listB: Union[int, List[int]]) -> bool:
        """ Check whether all the items in listB are in listA """
        if isinstance(listA, int):
            listA = [listA]
        if isinstance(listB, int):
            return listB in listA
        else:
            for item in listB:
                if item not in listA:
                    return False
            return True

    # Trim the drift terms
    for key in ham["drift"]:
        drifts = ham["drift"][key]
        if allIn(onQubits, drifts["on_qubits"]):
            subHam["drift"][key] = copy.deepcopy(drifts)
            subHam["drift"][key]["on_qubits"] = mapping(subHam["drift"][key]["on_qubits"])

    # Trim the control terms
    for key in ham["control"]:
        ctrls = ham["control"][key]
        if allIn(onQubits, ctrls["on_qubits"]):
            subHam["control"][key] = copy.deepcopy(ctrls)
            subHam["control"][key]["on_qubits"] = mapping(subHam["control"][key]["on_qubits"])

    # Update the basic information
    subHam["circuit"]["qubits"] = subQubits

    return subHam


def clearWaves(ham: Dict[str, Any], names: Union[str, List[str]] = None) -> None:
    """
    Remove all waveforms in the specified control terms.
    If names is None, remove all waveforms in all control terms.

    :param ham: the Hamiltonian dictionary
    :param names: the name of this term
    :return: None
    """
    if names is None:
        for name in ham["control"].keys():
            ham["control"][name]["waveforms"] = []
    elif isinstance(names, str):
        ham["control"][names]["waveforms"] = []
    elif isinstance(names, list):
        for name in names:
            ham["control"][name]["waveforms"] = []
    else:
        assert False, "Variable names should be a list or int."


def addWave(ham: Dict[str, Any], name: str, t0: float, t: float = 0, f: Union[Callable, str] = None,
            para: Dict[str, Any] = None, seq: List[float] = None) -> None:
    """
    Add a waveform to a certain control term. Quanlse provides three different methods to add
    waveforms to the Hamiltonian. In the following lines, we will use three examples to illustrate the three examples:

    :param ham: the Hamiltonian dictionary
    :param name: the name of the control term
    :param t0: the start time of the pulse
    :param t: the duration of the pulse
    :param f: the function of the waveform which takes the form: ``f(t, para)``
    :param para: pulse parameters passed to ``waveFunc``
    :param seq: a list of pulse amplitudes
    :return: None

    **Example 1** (using preset wave functions):

    .. code-block:: python

            p = {"a": 1.1, "tau": 10, "sigma": 8}
            qham.addWave(ham, "ctrlx", t0=0, t=20, f="gaussian", para=p)


    **Example 2** (using user-defined wave functions):

    .. code-block:: python

            def UserWaveFunc(t, args):
                return args["a"] + args["b"] + args["c"]

            p = {"a": 1.1, "b": 5.3, "c": 3.2}
            qham.addWave(ham, "ctrlx", t0=0, t=20, UserWaveFunc, para=p)


    **Example 3** (using user-defined wave sequences):

    .. code-block:: python

            s = [0.1 for _ in range(100)]
            qham.addWave(ham, "ctrlx", t0=0, seq=s)
    """

    # Record necessary information of the wave
    waveData = makeWaveData(ham, name, t0=t0, t=t, f=f, para=para, seq=seq)
    ham["control"][name]["waveforms"].append(waveData)
    # Calculate the max time.
    _, _ = computeMaxTime(ham)


def addWaveData(ham: Dict[str, Any], waveData: Union[Dict[str, Any], List[Dict[str, Any]]]) -> None:
    """
    Add waveforms to the control terms by waveData (generated by ``Utils.Waveforms.makeWaveData()``).
    Quanlse provides this function to add waveforms by dictionary data. In this way, users can save the
    wave data as a dictionary or a JSON string and import the wave batch by one function.

    :param ham: the Hamiltonian dictionary
    :param waveData: wave data dictionary generated by ``Utils.Waveforms.makeWaveData()``, or a list of wave data
    """

    if isinstance(waveData, list):
        for wave in waveData:
            name = wave["name"]
            addWave(ham, wave['name'], t0=wave['insert_ns'], t=wave['duration_ns'], f=wave['func'], para=wave['para'])
    elif isinstance(waveData, dict):
        wave = waveData
        addWave(ham, wave['name'], t0=wave['insert_ns'], t=wave['duration_ns'], f=wave['func'], para=wave['para'])
    else:
        assert False, "Only list and dict are allowed."


def setWave(ham: Dict[str, Any], name: str, t0: float, t: float, f: Union[Callable, str] = None,
            para: Dict[str, Any] = None, seq: List[float] = None) -> None:
    """
    Remove all waveforms, and add a waveform for a certain control term.

    :param ham: the Hamiltonian dictionary
    :param name: the name of the control term
    :param t0: the start time of the pulse
    :param t: the duration of the pulse
    :param f: the function of the waveform which takes the form: ``f(t, para)``
    :param para: pulse parameters passed to ``waveFunc``
    :param seq: a list of pulse amplitude
    :return: None

    **Example 1** (using preset wave functions):

    .. code-block:: python

            p = {"a": 1.1, "tau": 10, "sigma": 8}
            qham.setWave(ham, "ctrlx", t0=0, t=20, f="gaussian", para=p)


    **Example 2** (using user-defined wave functions):

    .. code-block:: python

            def UserWaveFunc(t, args):
                return args["a"] + args["b"] + args["c"]

            p = {"a": 1.1, "b": 5.3, "c": 3.2}
            qham.setWave(ham, "ctrlx", t0=0, t=20, UserWaveFunc, para=p)


    **Example 3** (using user-defined wave sequences):

    .. code-block:: python

            s = [0.1 for _ in range(100)]
            qham.setWave(ham, "ctrlx", t0=0, seq=s)
    """
    # We first clear all pre-defined wave
    clearWaves(ham, name)
    # Then add wave
    addWave(ham, name, t0=t0, t=t, f=f, para=para, seq=seq)


def removeDrift(ham: Dict[str, Any], names: Union[str, List[str]]) -> None:
    """
    Remove a specified drift term from the Hamiltonian.

    :param ham: the Hamiltonian dictionary
    :param names: the names of the control terms
    :return: None
    """
    if isinstance(names, str):
        assert names in ham["drift"].keys(), "Term does not exist."
        # We first extract necessary information of the circuit from Hamiltonian dictionary
        ham["drift"].pop(names)
    else:
        for name in names:
            assert name in ham["drift"].keys(), "Term does not exist."
            # We first extract necessary information of the circuit from Hamiltonian dictionary
            ham["drift"].pop(name)


def removeCoupling(ham: Dict[str, Any], names: Union[str, List[str]]) -> None:
    """
    Remove a specified coupling term from Hamiltonian.

    :param ham: the Hamiltonian dictionary
    :param names: the names of the coupling terms
    :return: None
    """
    if isinstance(names, str):
        assert names in ham["drift"].keys(), "Term does not exist."
        # We first extract necessary information of the circuit from Hamiltonian dictionary
        ham["drift"].pop(names)
        ham["drift"].pop(f"{names}(hc)")
    else:
        for name in names:
            assert name in ham["drift"].keys(), "Term does not exist."
            # We first extract necessary information of the circuit from Hamiltonian dictionary
            ham["drift"].pop(name)
            ham["drift"].pop(f"{name}(hc)")


def removeControl(ham: Dict[str, Any], names: Union[str, List[str]]) -> None:
    """
    Remove a specified control term from the Hamiltonian.

    :param ham: the Hamiltonian dictionary
    :param names: the names of the control terms
    :return: None
    """
    if isinstance(names, str):
        assert names in ham["control"].keys(), "Term does not exist."
        # We first extract necessary information of the circuit from Hamiltonian dictionary
        ham["control"].pop(names)
    else:
        for name in names:
            assert name in ham["control"].keys(), "Term does not exist."
            # We first extract necessary information of the circuit from Hamiltonian dictionary
            ham["control"].pop(name)


def getPulseSequences(ham: Dict[str, Any], names: Union[str, List[str]]) -> Dict[str, List[float]]:
    """
    Return the pulse sequences of specified control terms.

    :param ham: the Hamiltonian dictionary
    :param names: the name list of the control terms or a name string
    :return: a dictionary containing the pulse sequences
    """

    if isinstance(names, str):
        buildSequenceCache(ham)
        sequence = ham["cache"]["sequence"][names]
        clearCache(ham)
        return sequence
    else:
        buildSequenceCache(ham)
        sequences = {}
        for name in names:
            seq = ham["cache"]["sequence"][name]
            sequences[name] = seq
        clearCache(ham)
        return sequences


def getPulseWave(ham: Dict[str, Any], names: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Return the waveform configuration of a specified control Hamiltonian.

    :param ham: the Hamiltonian dictionary
    :param names: the names of the control terms
    :return: a dictionary containing the details of a waveform configuration
    """
    if isinstance(names, str):
        return ham["control"][names]["waveforms"]
    else:
        waves = {}
        for name in names:
            wave = ham["control"][names]["waveforms"]
            waves[name] = wave
        return waves


def generatePulseSequence(ham: Dict[str, Any], name: Union[str, List[str]]) -> List[float]:
    """
    Generate the piecewise-constant pulse sequence according to the waveform configurations during
    the whole evolution time.

    :param ham: the Hamiltonian dictionary
    :param name: the key of control term
    :return: the list of sequence
    """

    assert isinstance(name, str), "In generatePulseSequence, para `names` only accepts str."

    dt = ham["circuit"]["dt"]
    maxDt = ham["circuit"]["max_time_dt"]

    ctrls = ham["control"][name]
    sequenceList = []
    # Traverse all the time slices.
    for nowDt in range(0, maxDt):
        currentAmp = 0
        nowNs = nowDt * dt + dt / 2
        # Traverse all the waveforms
        for waveForm in ctrls["waveforms"]:
            insertNs = waveForm["insert_ns"]
            endNs = waveForm["insert_ns"] + waveForm["duration_ns"]
            if insertNs <= nowNs < endNs:
                # Calculate the waveforms' amplitudes.
                waveFunc = waveForm["func"]
                if waveFunc is None:
                    seq = waveForm["sequence"]
                    realTime = nowNs - waveForm["insert_ns"]
                    if int(realTime / dt) >= len(seq):
                        currentAmp += seq[-1]
                    else:
                        currentAmp += seq[int(realTime / dt)]
                elif isinstance(waveFunc, str):
                    currentAmp += play(waveForm["func"], nowNs - waveForm["insert_ns"], waveForm["para"])
                elif callable(waveFunc):
                    currentAmp += waveFunc(nowNs - waveForm["insert_ns"], waveForm["para"])
                else:
                    assert False, "Unsupported type of func."
        sequenceList.append(currentAmp)

    return sequenceList


def generateOperator(onQubits: Union[int, List[int]], matrices: Union[numpy.ndarray, List[numpy.ndarray]],
                     sysLevel: int, qubitNum: int) -> numpy.ndarray:
    """
    Generate the operator in the complete Hilbert space of the system by taking tensor products.

    :param onQubits: the index list of the terms are acting upon
    :param matrices: the terms corresponding to the qubits listed in ``onQubits``
    :param sysLevel: the dimension of the Hilbert space of single qubit
    :param qubitNum: the number of qubits in this system
    :return: the matrix form of the operator
    """
    # We first define the identity matrix to fill un-assigned qubits
    idMat = numpy.identity(sysLevel, dtype=complex)
    if isinstance(onQubits, int):
        assert numpy.size(matrices) == (sysLevel, sysLevel), "Dimension of matrix does not match the system Level."
        # The operator is on only one qubit.
        if onQubits == 0:
            # This operator is on the first qubit.
            operator = matrices
            for i in range(1, qubitNum):
                operator = numpy.kron(operator, idMat)
        else:
            # This operator is not on the first qubit.
            operator = idMat
            for i in range(1, onQubits):
                operator = numpy.kron(operator, idMat)
            operator = numpy.kron(operator, matrices)
            for i in range(onQubits + 1, qubitNum):
                operator = numpy.kron(operator, idMat)
        return operator
    elif isinstance(onQubits, list):
        operator = []
        for i in range(qubitNum):
            if i == 0:
                # On the first qubit
                if i in onQubits:
                    matrixIndex = onQubits.index(i)
                    operator = matrices[matrixIndex]
                    operatorSize = numpy.shape(matrices[matrixIndex])
                    assert operatorSize == (sysLevel, sysLevel), \
                        f"Dim of input matrix {operatorSize} does not match with the system level ({sysLevel})."
                else:
                    operator = idMat
            else:
                # Not on the first qubit
                if i in onQubits:
                    matrixIndex = onQubits.index(i)
                    operatorSize = numpy.shape(matrices[matrixIndex])
                    assert operatorSize == (sysLevel, sysLevel), \
                        f"Dim of input matrix {operatorSize} does not match with the system level ({sysLevel})."
                    operator = numpy.kron(operator, matrices[matrixIndex])
                else:
                    operator = numpy.kron(operator, idMat)
        return operator

    else:
        assert False, "Variable onQubits should be a list or an int."


def plotWaves(ham: Dict[str, Any], names: Union[str, List[str]] = None,
              color: Union[str, List[str]] = None, dark: bool = False) -> None:
    """
    Print the waveforms of the control terms listed in ``names``.

    :param ham: the Hamiltonian dictionary
    :param names: the name or name list of the control term
    :param color: None or list of colors
    :param dark: the plot can be switched to dark mode if required by user
    :return: None

    In Quanlse, users can specify colors from:

    ``mint``, ``blue``, ``red``, ``green``, ``yellow``, ``black``, ``pink``, ``cyan``, ``purple``,
    ``darkred``, ``orange``, ``brown``, ``pink`` and ``teal``.

    The colors will repeat if there are more pulses than colors.
    """

    # Get the max time
    maxNs = ham["circuit"]["max_time_ns"]

    # print plot
    if names is None:
        names = list(ham["control"].keys())
    elif isinstance(names, str):
        names = [names]
    elif isinstance(names, list):
        pass
    else:
        assert False, "Variable names should be a list or str."

    # Keep track of figure numbers
    fig = 0

    # Create two empty lists
    x = []
    y = []
    yLabel = []
    colors = []
    colorIndex = 0
    buildSequenceCache(ham)
    for name in names:
        aList = numpy.array(ham["cache"]["sequence"][name])
        tList = numpy.linspace(0, maxNs, len(aList))
        y.append(list(aList))
        x.append(list(tList))
        yLabel.append('Amp (a.u.)')

        # Whether repetitive colors or all blue
        if color is None:
            colors.append('blue')
        else:
            colors.append(color[colorIndex])
            colorIndex += 1
            if colorIndex == len(color):
                colorIndex = 0
        fig += 1
    plotPulse(x, y, xLabel='Time (ns)', yLabel=yLabel, title=names, color=colors, dark=dark)
    plt.show()
    clearCache(ham)


def printHam(ham: Dict[str, Any], digits: int = 4) -> None:
    """
    Display the basic information of the Hamiltonian.

    :param ham: the Hamiltonian dictionary
    :param digits: the precision of the numbers printed
    :return: None
    """

    qubitNum = ham['circuit']['qubits']

    print(f"\n====================\n1. Basic information\n====================\n")
    print(f"Title: `{ham['file']['title']}`")
    print(f"Qubits: {qubitNum}")
    print(f"System energy level: {ham['circuit']['sys_level']}")
    print(f"Sampling interval: {ham['circuit']['dt']} ns")
    print(f"Circuit duration: {ham['circuit']['max_time_ns']} ns")
    print(f"Calculation steps: {ham['circuit']['max_time_dt']}")

    # Obtain the max length name
    maxNameLengthDrift = 0 if len(ham["drift"]) == 0 else max([len(key) for key in ham["drift"]])
    maxNameLengthControl = 0 if len(ham["control"]) == 0 else max([len(key) for key in ham["control"]])
    maxNameLength = str(max(max(maxNameLengthDrift, maxNameLengthControl), 10))

    # Print abstract of operator
    print(f"\n============\n2. Operators\n============\n")
    qubitFormat = "{0: <5}  {1: <7}  {2: <" + maxNameLength + "}  {3: <9}  {4: <6} {5: <6}"
    print(qubitFormat.format('-' * 5, '-' * 7, '-' * 10, '-' * 9, '-' * 6, '-' * 6))
    print(qubitFormat.format("Qubit", "Type", "Name", "On qubits", "Pulses", "Amp"))
    for qubit in range(qubitNum):
        print(qubitFormat.format('-' * 5, '-' * 7, '-' * 10, '-' * 9, '-' * 6, '-' * 6))
        for key in ham["drift"]:
            drifts = ham["drift"][key]
            if qubit in drifts["on_qubits"]:
                print(qubitFormat.format(qubit, "Drift", key, f"{drifts['on_qubits']}", 0, f"{drifts['amp']}"))
        for key in ham["control"]:
            ctrls = ham["control"][key]
            if qubit in ctrls["on_qubits"]:
                print(qubitFormat.format(qubit, "Control", key, f"{ctrls['on_qubits']}",
                                         len(ctrls['waveforms']), "-"))

    # Print abstract of waveforms
    def paraRound(para: Dict[str, Any]) -> Union[Dict[str, Any], None]:
        """ Reduce the length of pulse Parameters """
        if para is None:
            return None
        else:
            for key in para:
                para[key] = round(para[key], digits)
            return para

    print(f"\n============\n3. Waveforms\n============\n")
    qubitFormat = "{0: <9}  {1: <" + maxNameLength + "}  {2: <20} {3: <5}  {4: <7}  {5: <45}"
    print(qubitFormat.format('-' * 9, '-' * 10, '-' * 15, '-' * 5, '-' * 7, '-' * 45))
    print(qubitFormat.format("On qubits", "Control", "Waveform", "Start", "Duration", "Params (Sequences)"))
    for key in ham["control"]:
        ctrls = ham["control"][key]
        if len(ctrls['waveforms']) > 0:
            print(qubitFormat.format('-' * 9, '-' * 10, '-' * 15, '-' * 5, '-' * 7, '-' * 45))
        for wave in ctrls['waveforms']:
            waveName = ""
            wavePara = ""
            if wave['func'] is None:
                waveName = "Manual Sequence"
                wavePara = f"Sequence contains {len(wave['sequence'])} pieces"
            elif callable(wave['func']):
                waveName = "Manual Wave"
                wavePara = f"{paraRound(wave['para'])}"
            elif isinstance(wave['func'], str):
                waveName = wave['func']
                wavePara = f"{paraRound(wave['para'])}"
            print(qubitFormat.format(f"{ctrls['on_qubits']}", key, waveName, wave['insert_ns'],
                                     wave['duration_ns'], wavePara))


def computeMaxTime(ham: Dict[str, Any]) -> Tuple[float, float]:
    """
    Compute the time duration of the whole circuit according to the waves added.

    :param ham: the Hamiltonian dictionary
    :return: a tuple of time duration in Nano-second and dt (AWG sampling interval)
    """
    # Find the longest time
    maxNs = 0
    for key in ham["control"]:
        ctrls = ham["control"][key]
        for waveform in ctrls["waveforms"]:
            finalNs = waveform["insert_ns"] + waveform["duration_ns"]
            if maxNs < finalNs:
                maxNs = finalNs
    maxDt = floor(maxNs / ham["circuit"]["dt"])

    ham["circuit"]["max_time_dt"] = maxDt
    ham["circuit"]["max_time_ns"] = maxNs

    return maxNs, maxDt


def getUnitary(ham: Dict[str, Any]) -> numpy.ndarray:
    """
    Get the unitary operator corresponding to the evolution governed by the Hamiltonian.

    :param ham: the Hamiltonian dictionary
    :return: the unitary operator
    """
    result = simulate(ham)
    return result["unitary"]


def simulate(ham: Dict[str, Any], recordEvolution: bool = False, jobList: List[List[Dict[str, Any]]] = None) \
        -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Calculate the unitary evolution operator with a given Hamiltonian. This function supports single-job and batch-job
    processing.

    :param ham: the Hamiltonian dictionary
    :param recordEvolution: the detailed procedure will be recorded if True
    :param jobList: a job list containing the waveform list
    :return: result dictionary (or a list of result dictionaries when ``jobList`` is provided)

    This function provides the option of simulating on local devices. However, Quanlse provides cloud computing
    services for faster simulation.

    1. If ``jobList`` is None, the waveforms in ham will be used for simulation, and return a result dictionary.

    **Example 1** (single-job processing):

    .. code-block:: python

            result = qham.simulate(ham)

    2. If ``jobList`` is not None, simulation will use the waveform list for batch-job processing provided in
       jobList, and return a list of result dictionaries.

    **Example 2** (batch-job processing):

    .. code-block:: python

            jobs = []
            for amp in ampList:
                jobWaves = []
                jobWaves.append(makeWaveData(ham, "ctrlx", t0=0, t=gateTime, f="gaussian", para=px))
                jobWaves.append(makeWaveData(ham, "ctrly", t0=0, t=gateTime, f="gaussian", para=py))
                jobs.append(jobWaves)

            result = qham.simulate(ham, jobList=jobs)

    Function ``Hamiltonian.makeWaveData()`` returns a dictionary containing the waveform configuration.
    """

    def _simulate(_ham: Dict[str, Any], _recordEvolution: bool = False) -> Dict[str, Any]:
        """ Simulate a Ham """
        # Start timing
        startTime = time.time()
        # Build cache
        buildCache(_ham)
        # Some constants
        dt = _ham["circuit"]["dt"]
        qubitNum = _ham["circuit"]["qubits"]
        sysLevel = _ham["circuit"]["sys_level"]
        # Obtain the drift Hamiltonian
        drift = _ham["cache"]["matrix_of_drift"]
        # Get the max time
        maxDt = _ham["circuit"]["max_time_dt"]
        # Start timing
        unitary = numpy.identity(sysLevel ** qubitNum, dtype=complex)
        unitaryList = []
        for nowDt in range(0, maxDt):
            totalHam = drift.copy()
            # Traverse all the control Hamiltonian
            for key in _ham["control"]:
                totalHam += _ham["cache"]["operator"]["control"][key] \
                            * _ham["cache"]["sequence"][key][nowDt]
            tUnitary = expm(- 1j * totalHam * dt)
            unitary = numpy.dot(tUnitary, unitary)
            if _recordEvolution:
                unitaryList.append(unitary)
        # We record necessary information of the unitary matrix
        ret = {
            "unitary": unitary,
            "dt": dt,
            "dts": maxDt,
            "ns": maxDt * dt,
            "time_consuming": time.time() - startTime,
            "evolution_history": unitaryList,
            "simulator": "built-in"
        }
        clearCache(_ham)
        return ret

    if jobList is None:
        return _simulate(ham, _recordEvolution=recordEvolution)
    else:
        # Initialize the result list
        resultList = []

        # Transverse all the jobs
        for job in jobList:
            # Clear all waves
            clearWaves(ham)
            # Add waves in current job
            for wave in job:
                if wave['sequence'] is not None and wave['func'] is None:
                    addWave(ham, wave['name'], t0=wave['insert_ns'], t=wave['duration_ns'], seq=wave['sequence'])
                elif wave['sequence'] is None and wave['func'] is not None:
                    addWave(ham, wave['name'], t0=wave['insert_ns'], t=wave['duration_ns'],
                            f=wave['func'], para=wave['para'])
                else:
                    assert False, "Waveform settings are not complete."

            # Start calculation
            buildCache(ham)
            result = simulate(ham, recordEvolution=False)
            clearCache(ham)
            resultList.append(result)
        return resultList


def clearCache(ham: Dict[str, Any]) -> None:
    """
    Initialize or clear the cache.

    :param ham: the Hamiltonian dictionary
    :return: None
    """

    ham["cache"] = {
        "matrix_of_drift": [],
        "operator": {
            "drift": {},
            "control": {}
        },
        "sequence": {}
    }


def buildCache(ham: Dict[str, Any]) -> None:
    """
    Generate the cache for further usage.
    In this function, we run ``buildOperatorCache()`` and ``buildSequenceCache()`` to build the
    global operator matrices and the pulse sequences before saving them to cache.

    :param ham: the Hamiltonian dictionary
    :return: None
    """

    # Initialize the Hamiltonian
    clearCache(ham)

    # Build operators and sequences
    buildOperatorCache(ham)
    buildSequenceCache(ham)


def buildOperatorCache(ham: Dict[str, Any]) -> None:
    r"""
    Save the drift/coupling/control terms in a more efficient way for further usage. Users can read the following
    lines on processed operator:

    :param ham: the Hamiltonian dictionary
    :return: None

    .. code-block:: python

            # Control terms:
            # buildOperatorCache will automatically run the tensor product procedure to produce control operators in
            # the full Hilbert space, hence the operators can be used directly in simulation:
            ctrlx = ham["cache"]["operator"]["control"]["ctrlx"]

            # Drift terms:
            # buildOperatorCache will not only take the tensor product of all drift terms, but also sums up
            # all the drift Hamiltonian into the one total drift Hamiltonian:
            totalDrift = ham["cache"]["operator"]["drift"]


    Note that this function will recursively process all the terms added to drift/coupling/control terms and
    save them to cache.
    """
    sysLevel = ham["circuit"]["sys_level"]
    qubitNum = ham["circuit"]["qubits"]

    # Generator the operator for all of the drift terms
    for key in ham["drift"]:
        drifts = ham["drift"][key]
        operator = generateOperator(drifts["on_qubits"], drifts["matrices"], sysLevel, qubitNum) * drifts["amp"]
        ham["cache"]["operator"]["drift"][key] = operator

    # Sum all the drift terms and save to the cache.
    driftTotal = numpy.zeros((sysLevel ** qubitNum, sysLevel ** qubitNum), dtype=complex)
    for key in ham["cache"]["operator"]["drift"]:
        driftTotal = driftTotal + ham["cache"]["operator"]["drift"][key]
    ham["cache"]["matrix_of_drift"] = driftTotal

    # Generator the pulse sequences for all of the control terms.
    for key in ham["control"]:
        ctrls = ham["control"][key]
        operator = generateOperator(ctrls["on_qubits"], ctrls["matrices"], sysLevel, qubitNum)
        ham["cache"]["operator"]["control"][key] = operator


def buildSequenceCache(ham: Dict[str, Any]) -> None:
    """
    Generate the pulse sequences for further usage.

    :param ham: the Hamiltonian dictionary
    :return: None
    """
    # Generator the pulse sequences for all of the control terms.
    for key in ham["control"]:
        sequence = generatePulseSequence(ham, key)
        ham["cache"]["sequence"][key] = sequence


def toJson(ham: Dict[str, Any]) -> str:
    """
    Transform the Hamiltonian dictionary to a string.

    :param ham: the Hamiltonian dictionary
    :return: a JSON formatted string
    """

    jham = copy.deepcopy(ham)
    clearCache(jham)

    maxDt = jham["circuit"]["max_time_dt"]
    dt = jham["circuit"]["dt"]

    # Check whether the waveFuncs are callable
    def wave2Seq(waveForm: Dict[str, Any]) -> List[float]:
        """ Translate callable waveform to sequence"""
        nonlocal maxDt, dt
        sequenceList = []
        # Traverse all the time slices.
        for nowDt in range(0, maxDt):
            nowNs = nowDt * dt + dt / 2
            insertNs = waveForm["insert_ns"]
            endNs = waveForm["insert_ns"] + waveForm["duration_ns"]
            if insertNs <= nowNs < endNs:
                # Calculate the waveforms' amplitudes.
                waveFunc = waveForm["func"]
                amp = waveFunc(nowNs - waveForm["insert_ns"], waveForm["para"])
            else:
                amp = 0.0
            sequenceList.append(amp)
        return sequenceList

    for key in jham["control"]:
        ctrls = jham["control"][key]
        for waveform in ctrls["waveforms"]:
            if waveform["func"] is not None and callable(waveform["func"]):
                print(f"Term {key} contains a callable waveform, it will be translated to a pulse sequence.")
                translatedSeq = wave2Seq(waveform)
                waveform["func"] = None
                waveform["para"] = None
                waveform["sequence"] = translatedSeq

    # Transform the control operators
    for key in jham["control"]:
        ctrls = jham["control"][key]
        # Modify the matrices
        if isinstance(ctrls["matrices"], list):
            mats = []
            for mat in ctrls["matrices"]:
                mats.append(numpyMatrixToDictMatrix(mat))
            ctrls["matrices"] = mats
        else:
            ctrls["matrices"] = numpyMatrixToDictMatrix(ctrls["matrices"])

    # Transform the drift operators
    for key in jham["drift"]:
        drifts = jham["drift"][key]
        # Modify the matrices
        if isinstance(drifts["matrices"], list):
            mats = []
            for mat in drifts["matrices"]:
                mats.append(numpyMatrixToDictMatrix(mat))
            drifts["matrices"] = mats
        else:
            drifts["matrices"] = numpyMatrixToDictMatrix(drifts["matrices"])

    return json.dumps(jham)


def createFromJson(jsonStr: str) -> Dict[str, Any]:
    """
    Transform a string (generated by ``toJson()``) to the Hamiltonian dictionary.

    :param jsonStr: the JSON formatted string
    :return: the Hamiltonian dictionary
    """
    jham = json.loads(jsonStr)
    clearCache(jham)

    # Transform the control operators
    for key in jham["control"]:
        ctrls = jham["control"][key]
        # Modify the matrices
        if isinstance(ctrls["matrices"], list):
            mats = []
            for mat in ctrls["matrices"]:
                mats.append(dictMatrixToNumpyMatrix(mat, complex))
            ctrls["matrices"] = mats
        else:
            ctrls["matrices"] = dictMatrixToNumpyMatrix(ctrls["matrices"], complex)

    # Transform the drift operators
    for key in jham["drift"]:
        drifts = jham["drift"][key]
        # Modify the matrices
        if isinstance(drifts["matrices"], list):
            mats = []
            for mat in drifts["matrices"]:
                mats.append(dictMatrixToNumpyMatrix(mat, complex))
            drifts["matrices"] = mats
        else:
            drifts["matrices"] = dictMatrixToNumpyMatrix(drifts["matrices"], complex)

    return jham
