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

r"""
Generally, when modeling a superconducting qubits system, we need to define the system's
Hamiltonian :math:`\hat{H}_{\rm sys}` firstly. The Hamiltonian, as the total energy, can
be used to describe the behavior of entire system:

:math:`\hat{H}_{\rm sys}(t) = \hat{H}_{\rm drift} + \hat{H}_{\rm coup} + \hat{H}_{\rm ctrl}(t)`

It typically contains three terms - the time-independent drift term describing the
individual qubits in the system, the coupling term describing the interaction between
qubits, and the time-dependent control term describing the control driving acting on qubits.

With the Hamiltonian constructed, we can simulate the evolution of the quantum system
by solving Schrödinger equation in the Heisenberg picture and then obtain the time-ordered
evolution operator :math:`U(t)`,

:math:`i\hbar\frac{{\rm \partial}U(t)}{{\rm \partial}t} = \hat{H}_{\rm sys}(t)U(t)`

A variety of functions and a complete set of pre-defined operators are provided in Quanlse,
which allows users to construct the Hamiltonian for large quantum systems with ease.

For more details, please visit https://quanlse.baidu.com/#/doc/tutorial-construct-ham.
"""

import copy
import pickle
import base64
from typing import Dict, Any, List, Union, Optional, Callable
from numpy import ndarray, shape, identity, size, kron, zeros, array, floor, transpose, conjugate
from numpy.linalg import eig

from Quanlse.QPlatform import Error
from Quanlse.QOperator import QOperator, create, destroy
from Quanlse.QWaveform import QWaveform, QJob, QJobList, QResult
from Quanlse.Utils.Functions import generateOperatorKey, tensor


class QHamiltonian:
    """
    Basic class of Hamiltonian, contains all information regarding a system.
    Users can create a QHamiltonian object to define the physical
    properties of a system. Constructing a Hamiltonian is essential to most Quanlse programs, we encourage the
    users to familiarize themselves with this step before advancing to the other features Quanlse provides.
    (see single-qubit gate tutorial for a complete example)

    :param subSysNum: size of the subsystem
    :param sysLevel: the energy levels of the system (support different levels for different qubits)
    :param dt: AWG sampling time
    :param title: a user-given title to the QHamiltonian object
    :param description: a user-given description
    """

    def __init__(self, subSysNum: int, sysLevel: Union[int, List[int]], dt: float,
                 title: str = "", description: str = "") -> None:
        """
        The constructor of the QHamiltonian class.
        """
        self.subSysNum = subSysNum  # type: int
        self.sysLevel = sysLevel  # type: Union[int, List[int]]
        self.title = title  # type: str
        self.description = description  # type: str
        self.dt = dt  # type: float

        self._driftOperators = {}  # type: Dict[str, Union[QOperator, List[QOperator]]]
        self._collapseList = None

        # Create QJob instance
        self._waveJob = QJob(subSysNum, sysLevel, dt)  # type: QJob
        self._waveJob.parent = self

        self._collapseOperators = None  # type: Optional[List[ndarray]]
        self._driftCache = None  # type: Optional[ndarray]
        self._ctrlCache = {}  # type: Optional[Dict[str, Any]]
        self._dissipationSuperCache = None  # type: Optional[ndarray]

        # Save the waveforms with flag QWAVEFORM_FLAG_DO_NOT_CLEAR
        self._doNotClearFlagOperators = {}  # type: Dict[str, Any]
        self._doNotClearFlagWaves = {}  # type: Dict[str, Any]

        # When extracting subsystem,
        #     this property keeps tracks the relations
        #     between the original and the new indices.
        self.subSystemIndexMapping = None  # type: Optional[Dict[int, int]]

    def __str__(self) -> str:
        """
        Printer function for the QHamiltonian object.
        """
        returnStr = ""

        # Print basic information
        returnStr += f"(1) Basic Information of QHam `{'None' if self.title == '' else self.title}`:\n"
        returnStr += f"    - System Level: {self.sysLevel}\n"
        returnStr += f"    - Number of subsystems: {self.subSysNum}\n"
        returnStr += f"    - dt: {self.dt}\n"
        returnStr += f"    - Description: {'None' if self.description == '' else self.description}\n"

        # Print drift terms
        returnStr += f"(2) Drift (Time-independent) Terms:\n"
        if len(self._driftOperators.keys()) < 1:
            returnStr += f"    No drift terms.\n"
        else:
            for key in self._driftOperators.keys():
                if isinstance(self._driftOperators[key], list):
                    returnStr += f"    - `{key}` * {self._driftOperators[key][0].coef}\n"
                else:
                    returnStr += f"    - `{key}` * {self._driftOperators[key].coef}\n"

        # Print control terms
        returnStr += f"(3) Control (Time-dependent) Terms and Waveforms:\n"
        if len(self._waveJob.ctrlOperators.keys()) < 1:
            returnStr += f"    No control terms.\n"
        else:
            for key in self._waveJob.ctrlOperators.keys():
                returnStr += f"    - `{key}`: {len(self._waveJob.waves[key])} wave(s).\n"

        # Print Local oscillator setting
        returnStr += f"(4) Local Oscillator Settings:\n"
        if len(self._waveJob.LO.keys()) < 1:
            returnStr += f"    No LO setting.\n"
        else:
            for key in self._waveJob.LO.keys():
                _freq = self._waveJob.LO[key][0]
                _phase = self._waveJob.LO[key][1]
                returnStr += f"    - `{key}`: frequency is {_freq}, phase is {_phase}.\n"

        return returnStr

    @property
    def subSysNum(self) -> int:
        """
        Return the number of subsystems.
        """
        return self._subSysNum

    @subSysNum.setter
    def subSysNum(self, value: int):
        """
        Set the number of subsystems.
        """
        if not isinstance(value, int):
            raise Error.ArgumentError("subSysNum must be an integer!")
        if value < 1:
            raise Error.ArgumentError("subSysNum must be larger than 0!")
        self._subSysNum = value

    @property
    def sysLevel(self) -> Union[int, List[int]]:
        """
        Return the level of subsystems.
        """
        return self._sysLevel

    @sysLevel.setter
    def sysLevel(self, value: Union[int, List[int]]):
        """
        Set the level of subsystems.

        :param value: the number of subsystems - either an integer value or a list of integers.
        """
        if not isinstance(value, int) and not isinstance(value, list):
            raise Error.ArgumentError("sysLevel must be an integer or a list!")
        if isinstance(value, list) and min(value) < 2:
            raise Error.ArgumentError("All items in sysSize must be larger than 1!")
        if isinstance(value, int) and value < 2:
            raise Error.ArgumentError("All items in sysSize must be larger than 1!")
        self._sysLevel = value

    @property
    def ctrlCache(self) -> Dict[str, Any]:
        """
        Return the cache of the control terms.
        """
        return self._ctrlCache

    @ctrlCache.deleter
    def ctrlCache(self):
        """
        Delete the cache of the control terms.
        """
        del self._ctrlCache

    @property
    def driftCache(self) -> ndarray:
        """
        Return the cache of the drift terms.
        """
        return self._driftCache

    @driftCache.deleter
    def driftCache(self):
        """
        Delete the cache of the drift terms.
        """
        del self._driftCache

    @property
    def waveCache(self) -> Dict[str, Any]:
        """
        Return the cache of drift terms.
        """
        return self._waveJob.waveCache

    @waveCache.deleter
    def waveCache(self):
        del self._waveJob.waveCache

    @property
    def driftOperators(self) -> Dict[str, Union[QOperator, List[QOperator]]]:
        """
        Return the operators of the drift terms.
        """
        return self._driftOperators

    @driftOperators.deleter
    def driftOperators(self):
        """
        Delete the operators of the drift terms.
        """
        del self._driftOperators
        self._driftOperators = {}

    @property
    def doNotClearFlagOperators(self) -> Dict[str, Union[QOperator, List[QOperator]]]:
        """
        Return operators with flag QWAVEFORM_FLAG_DO_NOT_CLEAR.
        """
        return self._doNotClearFlagOperators

    @property
    def doNotClearFlagWaves(self) -> Dict[str, Union[QWaveform, List[QWaveform]]]:
        """
        Return the waves with flag QWAVEFORM_FLAG_DO_NOT_CLEAR.
        """
        return self._doNotClearFlagWaves

    @property
    def ctrlOperators(self) -> Dict[str, Union[QOperator, List[QOperator]]]:
        """
        Return operators of the control term.
        """
        return self._waveJob.ctrlOperators

    @property
    def collapseOperators(self) -> Optional[List[ndarray]]:
        """
        Return the list of the cache of the collapse operators for the Lindblad master equations.
        """
        return self._collapseOperators

    @collapseOperators.deleter
    def collapseOperators(self):
        """
        Delete the list of the cache of the collapse operators for the Lindblad master equations.
        """
        del self._collapseOperators
        self._collapseOperators = None

    @property
    def dissipationSuperCache(self):
        """
        Return the cache of dissipation super operator.
        """
        return self._dissipationSuperCache

    @dissipationSuperCache.deleter
    def dissipationSuperCache(self):
        """
        Delete the cache of the dissipation super operators.
        """
        del self._dissipationSuperCache

    @property
    def job(self) -> QJob:
        """
        Return a QJob object. The QJob object contains all the information regarding a quantum task.
        """
        return self._waveJob

    @job.setter
    def job(self, job: QJob):
        """
        Set a QJob object.

        :param job: a QJob object
        """
        # Check if the input waveList matches the system conf of the QHam.
        if job.sysLevel is not None and job.sysLevel != self.sysLevel:
            raise Error.ArgumentError(f"sysLevel does not match ({job.sysLevel} and {self.sysLevel})!")
        if job.subSysNum != self.subSysNum:
            raise Error.ArgumentError(f"subSysNum does not match ({job.subSysNum} and {self.subSysNum})!")
        if not self.verifyWaveListObj(job):
            raise Error.ArgumentError("The input QJob is invalid for this ham, please check the "
                                      "system size or the dimension of the matrices.")
        # Save WaveList
        del self._waveJob
        self._waveJob = job
        self._waveJob.parent = self

    @job.deleter
    def job(self):
        """
        Delete a QJob object.
        """
        del self._waveJob

    def copy(self) -> "QHamiltonian":
        """
        Return a copy of current object.
        """
        return copy.deepcopy(self)

    def createJobList(self) -> QJobList:
        """
        Return an instance of a QJobList of the same system properties.
        """
        newJobList = QJobList(subSysNum=self.subSysNum, sysLevel=self.sysLevel, dt=self.dt)
        newJobList.LO = self.job.LO
        return newJobList

    def createJob(self) -> QJob:
        """
        Return an instance of a QJob object of the same system properties.
        """
        newJob = QJob(subSysNum=self.subSysNum, sysLevel=self.sysLevel, dt=self.dt)
        newJob.ctrlOperators = copy.deepcopy(self.job.ctrlOperators)
        newJob.LO = self.job.LO
        return newJob

    def addDrift(self, operator: Union[QOperator, Callable, List[QOperator], List[Callable]],
                 onSubSys: Union[int, List[int]], coef: float = None, name: str = None) -> None:
        r"""
        Add drift terms to the Hamiltonian.

        **Example 1** (add drift terms):

        Given a three-qubit system, the users could run the following code:

        .. code-block:: python

                ham.addDrift(operator=[number(2), number(2)], onSubSys=[0, 2], coef=omega1, name="Operator1")

        The code above allows the users to efficiently add the specified drift operator to the Hamiltonian:
        :math:`H = \omega_1 (\hat{a}^{\dagger}\hat{a}) \otimes \hat{I} \otimes (\hat{a}^{\dagger}\hat{a}),`

        where :math:`\hat{a},\hat{a}^{\dagger}` are the annihilation and creation operators; :math:`\hat{I}`
        is the identity operator.

        :param operator: the respective operator(s)
        :param onSubSys: qubit indexes that the term acts upon
        :param coef: the corresponding coefficient of the term
        :param name: a user-given name to the term for identifying purposes
        :return: None
        """

        _operators = []
        if isinstance(operator, list):
            for opIdx, op in enumerate(operator):
                if isinstance(op, QOperator):
                    # Input operator is an QOperator instance
                    _operators.append(copy.deepcopy(op))
                else:
                    # Input operator is callable
                    if isinstance(self.sysLevel, list):
                        _operators.append(op(self.sysLevel[onSubSys[opIdx]]))
                    else:
                        _operators.append(op(self.sysLevel))
        else:
            if isinstance(operator, QOperator):
                # Input operator is an QOperator instance
                _operators = copy.deepcopy(operator)
            else:
                # Input operator is callable
                if isinstance(self.sysLevel, list):
                    _operators = operator(self.sysLevel[onSubSys])
                else:
                    _operators = operator(self.sysLevel)

        if onSubSys is None:
            # Input the complete matrices directly.
            if isinstance(self._sysLevel, int):
                dim = self._sysLevel ** self._subSysNum
            elif isinstance(self._sysLevel, list):
                dim = 1
                for lvl in self._sysLevel:
                    dim = dim * lvl
            else:
                raise Error.ArgumentError(f"Wrong type of self._sysLevel ({type(self._sysLevel)}).")

            if not shape(_operators.matrix) == (dim, dim):
                raise Error.ArgumentError(f"Dimension of operator ({dim}, {dim}) does not match the sysLevel.")

            operatorForSave = _operators
        else:
            # Verify if operator and onSubSys have the same size
            if (isinstance(_operators, list) and not isinstance(onSubSys, list)) or \
                    (not isinstance(_operators, list) and isinstance(onSubSys, list)):
                raise Error.ArgumentError("The type of operator should be the same as onSubSys's!")

            if isinstance(_operators, list) and isinstance(onSubSys, list):
                if len(_operators) != len(onSubSys):
                    raise Error.ArgumentError(f"The size of operator ({len(_operators)}) != that "
                                              f"of onSubSys ({len(onSubSys)})!")

            # Verify the range and set the onSubSys
            if isinstance(onSubSys, int):
                if onSubSys >= self._subSysNum:
                    raise Error.ArgumentError(f"onSubSys ({onSubSys}) is larger than the "
                                              f"subSysNum {self._subSysNum}.")
                _operators.onSubSys = onSubSys
                operatorForSave = _operators
            elif isinstance(onSubSys, list):
                if max(onSubSys) >= self._subSysNum:
                    raise Error.ArgumentError(f"onSubSys ({onSubSys}) is larger than the "
                                              f"subSysNum {self._subSysNum}.")
                # Sort the operators according to on onSubSys
                sortedIndex = list(array(onSubSys).argsort())
                sortedOperators = [_operators[i] for i in sortedIndex]
                sortedOnSubSys = [onSubSys[i] for i in sortedIndex]
                # Set onSubSys
                for i in range(len(sortedIndex)):
                    sortedOperators[i].onSubSys = sortedOnSubSys[i]
                operatorForSave = sortedOperators
            else:
                raise Error.ArgumentError("onSubSys should be an integer or a list!")

        # Set the name if given
        if name is not None:
            opKey = name
        else:
            opKey = generateOperatorKey(self.subSysNum, operatorForSave)

        # Verify the uniqueness of the combination of the name and onSubSys
        if opKey in self._driftOperators.keys():
            raise Error.ArgumentError(f"Name ({opKey}) exists!")

        if coef is not None:
            if not isinstance(coef, float):
                raise Error.ArgumentError(f"coef should be float instead of {type(coef)}")
            if isinstance(_operators, list):
                for op in _operators:
                    op.coef = coef
            else:
                _operators.coef = coef

        self._driftOperators[opKey] = operatorForSave

    def addCoupling(self, onSubSys: List, g: float) -> None:
        r"""
        Add a coupling term to the Hamiltonian.

        :param onSubSys: the index list that the term works upon
        :param g: coupling strength
        :return: None.

        Two qubit indexes should be provided; and the coupling term is added between the two:

        :math:`H_{\rm coupling} = g (\hat{a}_{\rm q1}^{\dagger} \hat{a}_{\rm q2} +
        \hat{a}_{\rm q1} \hat{a}_{\rm q2}^{\dagger}),`

        where :math:`\hat{a}_q,\hat{a}_q^{\dagger}` are the annihilation and creation
        operators on the :math:`q`-th qubit.

        **Example 1** (add coupling terms):

        Given a three-qubit system, the users could run the following code:

        .. code-block:: python

                ham.addCoupling(onSubSys=[0, 2], g=0.0227 * 2 * pi)

        The code above allows the users to efficiently add the specified drift operator to the Hamiltonian:

        :math:`H = 0.14262830631 (\hat{a}_0 \otimes \hat{I} \otimes \hat{a}_2^{\dagger} + \hat{a}_0^{\dagger}
        \otimes \hat{I} \otimes \hat{a}_2).`
        """
        if len(onSubSys) != 2:
            raise Error.ArgumentError("Coupling term should be defined on subsystems of two qubits!")

        if isinstance(self._sysLevel, int):
            d = self._sysLevel
            # Construct matrices
            matrices = [destroy(d), create(d)]
            matricesHc = [create(d), destroy(d)]
        else:
            matrices = [destroy(self._sysLevel[onSubSys[0]]), create(self._sysLevel[onSubSys[1]])]
            matricesHc = [create(self._sysLevel[onSubSys[0]]), destroy(self._sysLevel[onSubSys[1]])]
        # Add drift terms
        self.addDrift(matrices, onSubSys=onSubSys, coef=g)
        self.addDrift(matricesHc, onSubSys=onSubSys, coef=g)

    def addWave(self, operators: Union[QOperator, Callable, List[QOperator], List[Callable]] = None,
                onSubSys: Union[int, List[int]] = None, waves: Union[QWaveform, List[QWaveform]] = None,
                t0: Union[int, float] = None, strength: Union[int, float] = 1.0,
                freq: Optional[Union[int, float]] = None, phase: Optional[Union[int, float]] = None,
                phase0: Optional[Union[int, float]] = None, name: str = None, tag: str = None, flag: int = None) \
            -> None:
        r"""
        This method adds a control term to the Hamiltonian with a specified waveform.

        :param operators: wave operator
        :param onSubSys: what subsystem the wave is acting upon
        :param waves: a QWaveform object
        :param t0: start time
        :param strength: wave strength
        :param freq: wave frequency shift
        :param phase: pulse phase shift (will accumulate during the entire pulse execution)
        :param phase0: pulse phase shift (will not accumulate during the entire pulse execution)
        :param name: a user-given name to the wave
        :param tag: wave tag indicating purpose
        :param flag: wave flag
        :return: None
        """
        self._waveJob.addWave(operators, onSubSys, waves, t0, strength, freq, phase, phase0, name, tag, flag)

    def appendWave(self, operators: Union[QOperator, Callable, List[QOperator], List[Callable]] = None,
                   onSubSys: Union[int, List[int]] = None, waves: Union[QWaveform, List[QWaveform]] = None,
                   strength: Union[int, float] = 1.0, freq: Optional[Union[int, float]] = None,
                   phase: Optional[Union[int, float]] = None, phase0: Optional[Union[int, float]] = None,
                   name: str = None, tag: str = None, shift: float = 0.0, compact: bool = True):
        """
        This method appends control terms and waveforms to a QJob object. Unlike `addWave()`, this function will append
        the waveform in the end of the existing waveforms, hence ignore the `t0` parameter of rhe `QWaveform` object.

        :param operators: wave operator
        :param onSubSys: what subsystem the wave is acting upon
        :param waves: a QWaveform object
        :param strength: wave strength
        :param freq: wave frequency shift
        :param phase: pulse phase shift (will accumulate during the entire pulse execution)
        :param phase0: pulse phase shift (will not accumulate during the entire pulse execution)
        :param name: a user-given name to the wave
        :param tag: wave tag indicating purpose
        :param shift: time shift of the added wave
        :param compact: the added wave will be left aligned in specified control terms if True.
        :return: None
        """
        self._waveJob.appendWave(operators, onSubSys, waves, strength, freq, phase, phase0,
                                 name, tag, shift, compact)

    def clearWaves(self, operators: Union[QOperator, Callable, List[QOperator], List[Callable]] = None,
                   onSubSys: Union[int, List[int]] = None, names: Union[str, List[str]] = None,
                   tag: str = None) -> None:
        """
        This method removes all waveforms in the specified control terms.
        If names is None, this method will remove waveforms in all control terms.

        :param operators: the corresponding operator(s)
        :param onSubSys: qubit indexes that the term acts upon
        :param names: the user-given name for identifying purposes
        :param tag: a user-given tag marking the pulse's usage
        :return: None.
        """
        self.job.clearWaves(operators, onSubSys, names, tag)

    def setLO(self, operators: Union[QOperator, Callable, List[QOperator], List[Callable]],
              onSubSys: Union[int, List[int]], name: str = None, freq: float = 1.0, phase: float = 0.0):
        """
        Add local oscillator for specific control terms.

        :param operators: QOperator object(s)
        :param onSubSys: what subsystem the wave is acting upon
        :param name: user-defined operator name
        :param freq: the frequency of the local oscillator
        :param phase: the phase between the local oscillator and the waveform
        """
        self._waveJob.setLO(operators, onSubSys, name, freq, phase)

    def plot(self, names: Union[str, List[str]] = None, tag: str = None, xUnit: str = 'ns', yUnit: str = 'Amp (a.u.)',
             color: Union[str, List[str]] = None, dark: bool = False) -> None:
        """
        Plot waves.

        :param names: "names" to plot
        :param tag: a tag marking the pulse's usage
        :param xUnit: x axis unit to be displayed
        :param yUnit: y axis unit to be displayed
        :param color: a list of colors for each pulse
            (from mint, blue, red, green, yellow, black, pink, cyan, purple, darkred, orange, brown, pink and teal)
            If pulse consists more than 250 cuts, switch to slice
        :param dark: enables a dark-themed mode
        :return: None.
        """
        self.job.plot(names, tag, xUnit, yUnit, color, dark)

    def setCollapse(self, cList: Optional[Dict[int, List[ndarray]]]) -> None:
        """
        Define the list of collapse operators for the Lindblad equation.

        :param cList: a list of collapse operators
        :return: None.
        """

        for index, operators in cList.items():
            if type(operators) is not list:
                raise Error.ArgumentError(f"the value of the key in cList should be a list type object")
            for operator in operators:
                if isinstance(self.sysLevel, int):
                    if shape(operator)[0] != self.sysLevel:
                        raise Error.ArgumentError("the dimension of the collapse operator doesn't equal"
                                                  " to the dimension of the state!")
                if isinstance(self.sysLevel, list):
                    if shape(operator)[0] != self.sysLevel[index]:
                        raise Error.ArgumentError("the dimension of the collapse operator doesn't equal"
                                                  " the dimension of the state!")
        self._collapseList = cList

    def verifyWaveListObj(self, waveJob: QJob) -> bool:
        """
        Check whether the QWaveList object is valid for this QHam object.

        :param waveJob: QWaveList to be checked
        :return: true of false indicating its validity
        """
        for key in waveJob.ctrlOperators.keys():
            if isinstance(waveJob.ctrlOperators[key], list):
                ops = waveJob.ctrlOperators[key]
            elif isinstance(waveJob.ctrlOperators[key], QOperator):
                ops = [waveJob.ctrlOperators[key]]
            else:
                raise Error.ArgumentError("The input QWaveList object is invalid!")
            # Check every operator
            for op in ops:
                if op.onSubSys >= self._subSysNum:
                    return False
                if isinstance(self._sysLevel, int):
                    if op.matrix is None:
                        return True
                    elif max(op.matrix.shape) > self._sysLevel:
                        return False
                else:
                    if max(op.matrix.shape) > self._sysLevel[op.onSubSys]:
                        return False
        return True

    def buildCache(self) -> None:
        """
        Build Cache for waveList and operators.

        :return: None
        """
        self.clearCache()
        self.buildOperatorCache()
        self._waveJob.buildWaveCache()
        if self._collapseList is not None:
            self.buildCollapseCache()
            self.buildDissipationSuperCache()

    def buildOperatorCache(self) -> None:
        r"""
        Save the drift/coupling/control terms in a more efficient way for further usage.
        Note that this function will recursively process all the terms added to drift/coupling/control terms and
        save them to cache.

        :return: None
        """
        sysLevel = self.sysLevel
        subSysNum = self.subSysNum

        # Generator the operator for all of the drift terms
        driftMatList = []
        for key in self._driftOperators.keys():
            if isinstance(self._driftOperators[key], list):
                mat = self._driftOperators[key][0].coef * self._generateOperator(self._driftOperators[key])
                driftMatList.append(mat)
            else:
                mat = self._driftOperators[key].coef * self._generateOperator(self._driftOperators[key])
                driftMatList.append(mat)
        # Sum all the drift terms and save to the cache.
        if isinstance(sysLevel, int):
            driftTotal = zeros((sysLevel ** subSysNum, sysLevel ** subSysNum), dtype=complex)
        else:
            dim = 1
            for i in sysLevel:
                dim = dim * i
            driftTotal = zeros((dim, dim), dtype=complex)
        for mat in driftMatList:
            driftTotal = driftTotal + mat
        self._driftCache = driftTotal

        # Generator the pulse sequences for all of the control terms.
        for key in self._waveJob.ctrlOperators.keys():
            ctrlsOp = self._waveJob.ctrlOperators[key]
            self._ctrlCache[key] = self._generateOperator(ctrlsOp)
        for key in self.doNotClearFlagOperators:
            if key not in self._ctrlCache:
                ctrlsOp = self.doNotClearFlagOperators[key]
                self._ctrlCache[key] = self._generateOperator(ctrlsOp)

    def buildCollapseCache(self) -> None:
        r"""
        Generate and save the cache of the collapse operators.

        :return: None
        """

        levelList = []

        if isinstance(self.sysLevel, list):
            levelList = self.sysLevel

        elif isinstance(self.sysLevel, int):
            levelList = [self.sysLevel for _ in range(self.subSysNum)]

        cList = self._collapseList

        cacheList = []

        for index, operators in cList.items():
            matList = [identity(level) for level in levelList]

            for operator in operators:
                matList[index] = operator
                mat = tensor(matList)
                cacheList.append(mat)

        self._collapseOperators = cacheList

    def buildDissipationSuperCache(self) -> None:
        """
        Generate the dissipation super operator in the Liouville form.
        """
        cList = self._collapseOperators

        dim = None

        # Calculate the dimension of the Hilbert space
        if isinstance(self.sysLevel, int):
            dim = self.sysLevel ** self.subSysNum

        elif isinstance(self.sysLevel, list):
            dim = 1
            for i in self.sysLevel:
                dim = dim * i

        idn = identity(dim, dtype=complex)

        ld = zeros((dim * dim, dim * dim), dtype=complex)

        if cList is not None:
            cdagList = [transpose(conjugate(c)) for c in cList]
            for c, cdag in zip(cList, cdagList):
                ld += kron(c, c) - 0.5 * kron(cdag @ c, idn) - 0.5 * kron(idn, cdag @ c)

        self._dissipationSuperCache = ld

    def _generateOperator(self, operator: Union[QOperator, List[QOperator]]) -> ndarray:
        """
        Generate the operator of the system in the complete Hilbert space by taking tensor products.

        :param operator: a list of operator(s)
        :return: the operator after taking the tensor products of the input operator
        """
        sysLevel = self.sysLevel
        subSysNum = self.subSysNum
        # Generate matrices
        if isinstance(operator, list):
            matrices = []
            for op in operator:
                matrices.append(op.matrix)
        else:
            matrices = operator.matrix
        # Each subsystem of the system has the same energy level.
        if isinstance(sysLevel, int):
            # We first define the identity matrix to fill un-assigned subsystems
            idMat = identity(sysLevel, dtype=complex)
            if isinstance(operator, QOperator):
                if size(matrices) == (sysLevel, sysLevel):
                    raise Error.ArgumentError("Dimension of matrix does not match the system Level.")
                # The operator is on only one subsystem.
                if operator.onSubSys == 0:
                    # This operator is on the first subsystem.
                    finalOperator = matrices
                    for i in range(1, subSysNum):
                        finalOperator = kron(finalOperator, idMat)
                else:
                    # This operator is not on the first subsystem.
                    finalOperator = idMat
                    for i in range(1, operator.onSubSys):
                        finalOperator = kron(finalOperator, idMat)
                    finalOperator = kron(finalOperator, matrices)
                    for i in range(operator.onSubSys + 1, subSysNum):
                        finalOperator = kron(finalOperator, idMat)
                return finalOperator
            elif isinstance(operator, list):
                finalOperator = []
                onSubSys = []
                for op in operator:
                    onSubSys.append(op.onSubSys)
                for i in range(subSysNum):
                    if i == 0:
                        # On the first subsystem
                        if i in onSubSys:
                            matrixIndex = onSubSys.index(i)
                            finalOperator = matrices[matrixIndex]
                            operatorSize = shape(matrices[matrixIndex])
                            if not (operatorSize == (sysLevel, sysLevel)):
                                raise Error.ArgumentError(f"Dim of input matrix {operatorSize} does not match"
                                                          f" with the system level ({sysLevel}, {sysLevel}).")
                        else:
                            finalOperator = idMat
                    else:
                        # Not on the first subsystem
                        if i in onSubSys:
                            matrixIndex = onSubSys.index(i)
                            operatorSize = shape(matrices[matrixIndex])
                            if not (operatorSize == (sysLevel, sysLevel)):
                                raise Error.ArgumentError(f"Dim of input matrix {operatorSize} does not match"
                                                          f" with the system level ({sysLevel}, {sysLevel}).")
                            finalOperator = kron(finalOperator, matrices[matrixIndex])
                        else:
                            finalOperator = kron(finalOperator, idMat)
                return finalOperator

            else:
                raise Error.ArgumentError("Variable onSubSys should be a list or an int value.")
        # The sysLevel is a list of different energy levels for multiple subsystems
        elif isinstance(sysLevel, list):
            # Create a list of identities of different dimension for each subsystem of different energy level
            idMat = [identity(i, dtype=complex) for i in sysLevel]
            # The operator is acting on only one subsystem.
            if isinstance(operator, QOperator):
                if not (matrices.shape == (sysLevel[operator.onSubSys], sysLevel[operator.onSubSys])):
                    raise Error.ArgumentError("Dimension of matrix does not match the system Level.")
                # The operator is acting on the first subsystem.
                if operator.onSubSys == 0:
                    finalOperator = matrices
                    for i in range(1, subSysNum):
                        finalOperator = kron(finalOperator, idMat[i])
                else:
                    # This operator is not acting on the first subsystem.
                    finalOperator = idMat[0]
                    for i in range(1, operator.onSubSys):
                        finalOperator = kron(finalOperator, idMat[i])
                    finalOperator = kron(finalOperator, matrices)
                    for i in range(operator.onSubSys + 1, subSysNum):
                        finalOperator = kron(finalOperator, idMat[i])
                return finalOperator
            # The operator is acting on multiple subsystems.
            elif isinstance(operator, list):
                finalOperator = []
                onSubSys = []
                for op in operator:
                    onSubSys.append(op.onSubSys)
                for i in range(subSysNum):
                    if i == 0:
                        # Acting on the first subsystem
                        if i in onSubSys:
                            matrixIndex = onSubSys.index(i)
                            finalOperator = matrices[matrixIndex]
                            operatorSize = shape(matrices[matrixIndex])
                            if not (operatorSize == (sysLevel[i], sysLevel[i])):
                                raise Error.ArgumentError(f"Dim of input matrix {operatorSize} does not match"
                                                          f" with the system level ({sysLevel[i]}).")
                        else:
                            finalOperator = idMat[i]
                    else:
                        # Not acting on the first subsystem
                        if i in onSubSys:
                            matrixIndex = onSubSys.index(i)
                            operatorSize = shape(matrices[matrixIndex])
                            if not (operatorSize == (sysLevel[i], sysLevel[i])):
                                raise Error.ArgumentError(f"Dim of input matrix {operatorSize} does not match"
                                                          f" with the system level ({sysLevel[i]}).")
                            finalOperator = kron(finalOperator, matrices[matrixIndex])
                        else:
                            finalOperator = kron(finalOperator, idMat[i])
                return finalOperator

            else:
                raise Error.ArgumentError("Variable onSubSys should be a list or an int value.")

    def clearCache(self):
        """
        Clear cache.
        """
        self._ctrlCache = {}
        self._driftCache = None
        self._collapseOperators = None
        self._waveJob.clearCache()

    def subSystem(self, onSubSys: Union[int, List[int]]) -> 'QHamiltonian':
        """
        This method extracts a subsystem from the target Hamiltonian.

        :param onSubSys: a list of qubit indexes
        :return: subsystem's QHamiltonian object
        """
        subHam = copy.deepcopy(self)
        subHam.clearCache()
        # Information about the sub system
        if isinstance(onSubSys, int):
            subQubits = 1
            indexMapping = {onSubSys: 0}
            if onSubSys >= self._subSysNum:
                raise Error.ArgumentError("onSubSys should be less than the subSysNum of this QHamiltonian.")
        elif isinstance(onSubSys, list):
            subQubits = len(onSubSys)
            # qubit index mapping
            indexMapping = {}
            for newId, originalId in enumerate(onSubSys):
                indexMapping[originalId] = newId
                if originalId >= self._subSysNum:
                    raise Error.ArgumentError("all onSubSys items should be less than the subSysNum "
                                              "of this QHamiltonian.")
        else:
            raise Error.ArgumentError("Unsupported input of qubitNum, it should be an int value or a list.")

        # Update the sysLevel property
        if isinstance(self._sysLevel, list):
            if isinstance(onSubSys, int):
                subHam.sysLevel = self._sysLevel[onSubSys]
                subHam.job.sysLevel = self._sysLevel[onSubSys]
            else:
                subHam.sysLevel = [self._sysLevel[qIndex] for qIndex in onSubSys]
                subHam.job.sysLevel = [self._sysLevel[qIndex] for qIndex in onSubSys]

        # Set the mapping relationship
        subHam.subSystemIndexMapping = copy.deepcopy(indexMapping)

        def mapping(index: Union[int, List[int]]) -> Union[int, List[int]]:
            """
            Map the `onQubits` index(es) from the original Ham to the extracted Ham.

            :param index: 'onQubits' index(es) from the original Ham
            :return: mapped index(es) onto the extracted Ham
            """
            if isinstance(index, int):
                return indexMapping[index]
            else:
                mappedList = []
                for item in index:
                    mappedList.append(indexMapping[item])
                return mappedList

        def allIn(listA: Union[int, List[int]], listB: Union[int, List[int]]) -> bool:
            """
            Check whether all the items in listB are in listA.

            :param listA: listA
            :param listB: listB
            :return: a bool value
            """
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
        del subHam.driftOperators
        for key in self._driftOperators:
            drifts = self._driftOperators[key]  # type: Union[QOperator, List[QOperator]]
            if isinstance(drifts, list):
                _coef = drifts[0].coef
                originalOnSubSys = [op.onSubSys for op in drifts]
            else:
                _coef = drifts.coef
                originalOnSubSys = copy.deepcopy(drifts.onSubSys)
            if allIn(onSubSys, originalOnSubSys):
                subHam.addDrift(copy.deepcopy(drifts), mapping(originalOnSubSys), coef=_coef)

        # Trim the control terms
        subHam.job.clearWaves()
        for key in self.job.ctrlOperators:
            ctrls = self.job.ctrlOperators[key]  # type: Union[QOperator, List[QOperator]]
            if isinstance(ctrls, list):
                originalOnSubSys = [op.onSubSys for op in ctrls]
            else:
                originalOnSubSys = ctrls.onSubSys
            if allIn(onSubSys, originalOnSubSys):
                wavesDict = self.job.searchWave(ctrls, originalOnSubSys)
                for waveKey in wavesDict.keys():
                    # Update the waveform setup
                    subHam.job.addWave(ctrls, mapping(originalOnSubSys), waves=wavesDict[waveKey])

        # Trim the DoNotClear terms
        subHam.doNotClearFlagWaves.clear()
        subHam.doNotClearFlagOperators.clear()

        for key in self.doNotClearFlagWaves:
            doNotClrCtrl = self.doNotClearFlagOperators[key]  # type: Union[QOperator, List[QOperator]]
            if isinstance(doNotClrCtrl, list):
                originalOnSubSys = [op.onSubSys for op in doNotClrCtrl]
            else:
                originalOnSubSys = doNotClrCtrl.onSubSys
            if allIn(onSubSys, originalOnSubSys):
                wavesDict = QJob.searchWaveTool(self.doNotClearFlagWaves, self.subSysNum, self.sysLevel,
                                                doNotClrCtrl, originalOnSubSys)
                for waveKey in wavesDict.keys():
                    # Update the waveform setup
                    subHam.addWave(doNotClrCtrl, mapping(originalOnSubSys), waves=wavesDict[waveKey])

        # Trim the collapse terms
        cList = copy.deepcopy(self._collapseList)
        subCList = None
        if cList is not None:
            subCList = {}
            for originalId, newId in indexMapping.items():
                if originalId not in cList.keys():
                    pass
                else:
                    subCList[newId] = cList[originalId]

        # Update the basic information
        subHam._collapseList = subCList
        subHam.subSysNum = subQubits
        subHam.job.subSysNum = subQubits

        return subHam

    def eigen(self, t: Optional[ndarray] = None):
        """
        Calculate the eigenvalues and eigenvectors of the given Hamiltonian.

        :param t: The time at which the Hamiltonian's eigenvectors and eigenvalues are to compute.
        :return: If t is none, it returns (n, n) Hamiltonian's eigenvalues of shape (n, ) in the
        ascending order and the corresponding eigenvectors matrix of shape (n, n); If t is an (m, ) array,
        it returns list of (m, n) eigenvalues and (m, n, n) eigenvectors.
        """

        # Build cache of the Hamiltonian
        self.buildCache()
        drift = self.driftCache
        ctrl = self.ctrlCache
        wave = self.job.waveCache
        dt = self.dt
        maxDt = self.job.endTimeDt

        def _computeEigen(matrix):
            # Solve eigenvalues problem
            eigenVals, eigenVecs = eig(matrix)
            # Rearrange eigenvalues and eigenvectors in the ascending order
            sortedIndex = eigenVals.argsort()
            eigenVecs = eigenVecs[:, sortedIndex]
            eigenVals = eigenVals[sortedIndex]
            return eigenVals.real, eigenVecs

        if t is None:
            # Solve eigenvalues problem
            vals, vecs = _computeEigen(drift)
        else:
            valsList = []
            vecsList = []
            if len(t) > maxDt:
                raise Error.ArgumentError('The length of time exceeds the number of maximum steps')
            indexDt = floor(t / dt)
            indexDt = indexDt.astype(int)
            for nowDt in indexDt:
                if nowDt >= maxDt:
                    nowDt = nowDt - 1
                ham = drift
                for key in ctrl:
                    ham += ctrl[key] * wave[key][nowDt]
                evals, evecs = _computeEigen(ham)
                valsList.append(evals.tolist())
                vecsList.append(evecs.tolist())
            vals = array(valsList)
            vecs = array(vecsList)

        # Clear cache
        self.clearCache()
        return vals, vecs

    def subSystemIndicesInverse(self) -> Dict[int, int]:
        """
        In self.subSystem, users can extract a subsystem of the original system,
        thus, the qubit indices will change. For example, for a system with indices [0, 1, 2, 3],
        we extract the subsystem constituted by subsystems 1 and 3, then the index of
        subsystem 1 changes to 0; 3 changes to 1; the mapping in self.subSystemIndexMapping is:
        {1: 0, 3: 1}.
        We can use this function to obtain the inverse mapping, the inverse mapping of the mapping
        above is:
        {0: 1, 1: 3}

        :return: a dictionary containing the 'inverse mapping'
        """
        inverseMapping = {}
        for key in self.subSystemIndexMapping:
            inverseMapping[self.subSystemIndexMapping[key]] = key
        return inverseMapping

    def outputInverseJob(self, subSysNum: int, sysLevel: int = None, dt: float = None) -> QJob:
        """
        Return an inverse-mapped Job. (see subSystemIndicesInverse())

        :param subSysNum: subsystem's size
        :param sysLevel: subsystem's energy level
        :param dt: time interval
        :return: returned QJob object
        """
        # Get the job
        inverseMapping = self.subSystemIndicesInverse()
        _sysLevel = self.sysLevel if sysLevel is None else sysLevel
        _dt = self.dt if dt is None else dt
        job = QJob(subSysNum, _sysLevel, _dt)
        for opKey in self.job.ctrlOperators.keys():
            operators = self.job.ctrlOperators[opKey]
            onSubSysInSubHam = self.job.ctrlOperators[opKey].onSubSys
            if isinstance(operators, list):
                onSubSysInOriginalHam = []
                for index in onSubSysInSubHam:
                    onSubSysInOriginalHam.append(inverseMapping[index])
            elif isinstance(operators, QOperator):
                onSubSysInOriginalHam = inverseMapping[onSubSysInSubHam]
            else:
                raise Error.ArgumentError("Unsupported type of operators.")
            # Add the waves into the new QJob instance.
            for wave in self.job.waves[opKey]:
                job.addWave(operators, onSubSysInOriginalHam, wave)
        return job

    def simulate(self, job: QJob = None, state0: ndarray = None, recordEvolution: bool = False, shot: int = None,
                 isOpen: bool = False, jobList: QJobList = None, refreshCache: bool = True, accelerate: str = None,
                 adaptive: bool = False, tolerance: float = 0.01) -> QResult:
        """
        Calculate the unitary evolution operator with a given Hamiltonian. This function supports
        both single-job and batch-job processing.

        To activate acceleration, please install ``Numba`` JIT compiler, please visit its official website
        https://numba.pydata.org/ for more details.

        :param job: the QJob object to simulate.
        :param state0: the initial state vector. If None is given, this function will return the time-ordered
                       evolution operator, otherwise returns the final state vector.
        :param recordEvolution: the detailed procedure will be recorded if True
        :param shot: return the population of the eigenstates when ``shot`` is provided
        :param isOpen: simulate the evolution using Lindblad equation
        :param jobList: a job list containing the waveform list
        :param refreshCache: it will neither clear nor rebuild cache if refreshCache is set to be True
        :param adaptive: use the adaptive solver if True
        :param tolerance: the greatest error of approximation for the adaptive solver
        :param accelerate: indicates the accelerator

        :return: result dictionary (or a list of result dictionaries when ``jobList`` is provided)

        This function does provide the option of simulating on local devices. However, Quanlse also provides
        cloud computing services which are significantly faster.

        **Example 1** (single-job processing):

        If ``jobList`` is None, the waveforms in ham will be used for simulation, and return a result dictionary.

        .. code-block:: python

                result = ham.simulate()

        **Example 2** (batch-job processing):

        If ``jobList`` is not None, simulation will use the waveform list for batch-job processing provided in
        jobList, and return a list of result dictionaries.

        .. code-block:: python

                result = ham.simulate(ham, jobList=jobs)

        Here, jobs is a QJobList object, indicating the the waveforms to be used.
        """

        resultList = QResult("built-in simulator")

        if jobList is None:
            jobList = self.createJobList()
            if job is None:
                jobList.addJob(self.job)
            else:
                jobList.addJob(job)

        # Transverse all the jobs
        _tmpHam = copy.deepcopy(self)
        for waveList in jobList.jobs:
            # Clear all waves
            _tmpHam.job.clearWaves()
            _tmpHam.job.LO = waveList.LO
            # Add waves in current job
            _tmpHam.job.appendJob(waveList)
            # Build cache
            if refreshCache:
                _tmpHam.buildCache()
            # Start calculation
            if isOpen:
                if not adaptive:
                    from Quanlse.Utils.ODESolver import solverOpenSystem
                    result = solverOpenSystem(_tmpHam, state0=state0, recordEvolution=recordEvolution,
                                              accelerate=accelerate)
                else:
                    raise Error.ArgumentError("Adaptive solver is not available when solving open system.")
            else:
                if not adaptive:
                    from Quanlse.Utils.ODESolver import solverNormal
                    result = solverNormal(_tmpHam, state0=state0, shot=shot, recordEvolution=recordEvolution,
                                          accelerate=accelerate)
                else:
                    from Quanlse.Utils.ODESolver import solverAdaptive
                    result = solverAdaptive(_tmpHam, state0=state0, shot=shot, tolerance=tolerance,
                                            accelerate=accelerate)
            resultList.append(result)
            if refreshCache:
                self.clearCache()
        return resultList

    def doNotClearWaveFunctionsToSequences(self, maxEndTime: float = None) -> Dict[str, List[QWaveform]]:
        """
        Convert all functions into sequence (including the QWaveform objects) that is serializable.

        :param maxEndTime: maximum ending time
        :return: dictionary containing the converted functions
        """
        newList = {}
        for waveKey in self._doNotClearFlagWaves:
            # Add QWaveform
            newList[waveKey] = []
            for wave in self._doNotClearFlagWaves[waveKey]:
                seqWave = wave.waveFunctionToSequence(self.dt, maxEndTime=maxEndTime)
                newList[waveKey].append(seqWave)
        return newList

    def dump(self, maxEndTime: float = None) -> str:
        """
        Return a base64 encoded string.

        :param maxEndTime: maximum ending time
        :return: encoded string
        """
        obj = copy.deepcopy(self)
        obj._waveJob = None

        if self._waveJob is not None:
            del obj._waveJob
            obj._waveJob = self._waveJob.waveFunctionsToSequences(maxEndTime=maxEndTime)

        if self._doNotClearFlagWaves is not None:
            obj._doNotClearFlagWaves = None
            obj._doNotClearFlagWaves = self.doNotClearWaveFunctionsToSequences(maxEndTime=maxEndTime)
            for opKey in obj._doNotClearFlagWaves.keys():
                for wave in obj._doNotClearFlagWaves[opKey]:
                    if not hasattr(wave, "omega"):
                        wave.omega = wave.freq
                        wave.phi = 0.
                    if not hasattr(wave, "phi"):
                        wave.phi = 0. if wave.phase0 is None else wave.phase0

        if obj.job is not None:
            obj.job.parent = None

        # For compatibility with previous versions
        for opKey in obj.job.waves.keys():
            for wave in obj.job.waves[opKey]:
                if not hasattr(wave, "omega"):
                    wave.omega = wave.freq
                    wave.phi = 0.
                if not hasattr(wave, "phi"):
                    wave.phi = 0. if wave.phase0 is None else wave.phase0

        byteStr = pickle.dumps(obj)
        base64str = base64.b64encode(byteStr)
        del obj
        return base64str.decode()

    def clone(self) -> 'QHamiltonian':
        """
        Return the copy of the object
        """
        return copy.deepcopy(self)

    @staticmethod
    def load(base64Str: str) -> 'QHamiltonian':
        """
        Create object from base64 encoded string.

        :param base64Str: a base64 encoded string
        :return: QHam object
        """
        byteStr = base64.b64decode(base64Str.encode())
        obj = pickle.loads(byteStr)  # type: QHamiltonian

        if obj.job is not None:
            # Set job's parent
            obj.job.parent = obj
            # For compatibility with previous versions
            for opKey in obj.job.waves.keys():
                for wave in obj.job.waves[opKey]:
                    if not hasattr(wave, "freq"):
                        wave.freq = wave.omega if hasattr(wave, "omega") else None
                    if not hasattr(wave, "phase0"):
                        wave.phase0 = wave.phi if hasattr(wave, "phi") else None
                    if not hasattr(wave, "phase"):
                        wave.phase = None
        return obj
