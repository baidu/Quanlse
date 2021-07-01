#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2020 Baidu, Inc. All Rights Reserved.
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
Simulating the time evolution of the qubits at the pulse level gives us more insight into how
quantum gates operates and the effects of noise. For superconducting quantum circuit, the
transmon qubits are controlled by applying microwave and magnetic flux. However, the performance
of quantum gates are often suppressed by various factors - the decoherence of the qubit due
to its interaction with the environment, the unwanted cross-talk effect and leakage into the
higher levels of the transmon.

The pulse-level simulator provided by Quanlse allows us to simulate quantum operations on
noisy quantum device consisting of multiple transmon qubits to better understand the physics
behind quantum computing.

For more details, please visit https://quanlse.baidu.com/#/doc/tutorial-pulse-level-simulator.
"""

import random
import copy
from typing import List, Union, Optional, Dict, Tuple, Iterable, Callable
from numpy import pi, sqrt, ndarray, zeros, identity, random

from Quanlse.QHamiltonian import QHamiltonian
from Quanlse.QPlatform import Error
from Quanlse.QOperator import QOperator, duff, create, destroy, a, adagger, number
from Quanlse.QWaveform import QJob, square, QWaveform, sequence, \
    QWAVEFORM_FLAG_ALWAYS_ON, QWAVEFORM_FLAG_DO_NOT_CLEAR, QWAVEFORM_FLAG_DO_NOT_PLOT
from Quanlse.Utils.Functions import tensor
from Quanlse.Scheduler.Superconduct import SchedulerSuperconduct


class PulseModel(SchedulerSuperconduct):
    """
    Basic class of PulseModel. It defines the system information and noise-related parameters. It is also a sub-class of
    SchedulerSuperconduct.

    :param subSysNum: size of the subsystem
    :param sysLevel: the energy levels of the system (support different levels for different qubits)
    :param qubitFreq: qubit frequencies, in 2 pi GHz
    :param qubitAnharm: Anharmonicity for each qubit
    :param dt: sampling time period:
    :param couplingMap: direct coupling structure
    :param T1: T1-relaxation time, in nanoseconds
    :param T2: T2-dephasing time, in nanoseconds
    :param ampSigma: deviation of the amplitude distortion
    :param betaStd: standard deviation of the relative strength in classical crosstalk
    :param frame: rotating frame
    :param pulseGenerator: customized pulse generator for scheduler
    """

    def __init__(self,
                 subSysNum: int,
                 sysLevel: Union[int, List[int]],
                 qubitFreq: Dict[int, Union[int, float]],
                 qubitAnharm: Dict[int, Union[int, float]],
                 dt: float = 0.2,
                 couplingMap: Optional[Dict[Tuple, Union[int, float]]] = None,
                 T1: Optional[Dict[int, Union[int, float, None]]] = None,
                 T2: Optional[Dict[int, Union[int, float, None]]] = None,
                 ampSigma: Optional[Union[int, float]] = None,
                 betaStd: Optional[Union[int, float]] = None,
                 frame: Optional[Dict[int, Union[int, float]]] = None,
                 pulseGenerator: Callable = None):

        self._subSysNum = subSysNum
        self._sysLevel = sysLevel
        self._dt = dt

        self.qubitFreq = qubitFreq  # type: Dict[int, Union[int, float]]
        self.qubitAnharm = qubitAnharm  # type: Dict[int, Union[int, float]]
        self.T1 = T1  # type: Optional[Dict[int, Union[int, float]]]
        self.T2 = T2  # type: Optional[Dict[int, Union[int, float]]]
        self.ampSigma = ampSigma  # type: Optional[Union[int, float]]
        self.betaStd = betaStd  # type: Optional[Union[int, float]]
        self.frame = frame  # type: Optional[Dict[int, Union[int, float]]]
        self._couplingMap = couplingMap  # type: Optional[Dict[Tuple, Union[int, float]]]

        _ham = self.createQHamiltonian()

        if pulseGenerator is not None:
            super(PulseModel, self).__init__(dt, ham=_ham, subSysNum=subSysNum,
                                             sysLevel=sysLevel, pulseGenerator=pulseGenerator(_ham))
        else:
            super(PulseModel, self).__init__(dt, ham=_ham, subSysNum=subSysNum,
                                             sysLevel=sysLevel, pulseGenerator=None)

    @property
    def dim(self) -> int:
        """
        Return the dimension of the Hilbert space.
        """
        _dim = 1
        if isinstance(self.sysLevel, int):
            _dim = self.sysLevel ** self.subSysNum

        elif isinstance(self.sysLevel, list):
            _dim = 1
            for level in self.sysLevel:
                _dim = _dim * level

        return _dim

    @property
    def qubitFreq(self) -> Dict[int, Union[int, float]]:
        """
        Return the eigenfrequency of each qubit in the system.
        """
        return self._qubitFreq

    @qubitFreq.setter
    def qubitFreq(self, value: Dict[int, Union[int, float]]):
        """
        Return the eigen frequency of each qubit in the system, in 2 * pi * GHz.
        """
        if len(value) != self.subSysNum:
            raise Error.ArgumentError("The number of the frequencies and the number of qubits are not the same.")
        self._qubitFreq = value

    @property
    def qubitAnharm(self) -> Dict[int, Union[int, float]]:
        """
        Return the anharmonicity of each qubit in the system.
        """
        return self._qubitAnharm

    @qubitAnharm.setter
    def qubitAnharm(self, value: Dict[int, Union[int, float]]):
        """
        Setter for qubits anharmonicity.
        """
        if len(value) != self.subSysNum:
            raise Error.ArgumentError("The number of the anharmonicities and the number of qubits are not the same.")
        self._qubitAnharm = value

    @property
    def couplingMap(self) -> Optional[Dict[Tuple, Union[int, float]]]:
        """
        Return the eigenfrequency of each qubit in the system.
        """
        return self._couplingMap

    @couplingMap.setter
    def couplingMap(self, value: Optional[Dict[Tuple, Union[int, float]]]):
        """
        Return the coupling structure of the system.
        """
        if self.couplingMap is not None:
            indexList = [index for pair in value for index in pair]
            for index in indexList:
                if index not in range(self.subSysNum+1):
                    raise Error.ArgumentError("The number of the frequencies and the number of qubits are not the same.")
        self._couplingMap = value

    @property
    def T1(self) -> Optional[Dict[int, Union[int, float, None]]]:
        """
        Return the relaxation time T1 for each qubit.
        """
        return self._T1

    @T1.setter
    def T1(self, value: Optional[Dict[int, Union[int, float, None]]]):
        """
        Setter for the relaxation time T1 for each qubit.
        """
        if value is not None:
            for index, t1 in value.items():
                if t1 < 0:
                    raise Error.ArgumentError("The value of T1 should be a positive real number.")
                if index not in range(self.subSysNum):
                    raise Error.ArgumentError(f"{index}th-qubit is not defined.")
        self._T1 = value

    @property
    def T2(self) -> Optional[Dict[int, Union[int, float, None]]]:
        """
        Return the relaxation time T2 for each qubit.
        """
        return self._T2

    @T2.setter
    def T2(self, value: Optional[Dict[int, Union[int, float, None]]]):
        """
        Setter for the relaxation time T2 for each qubit.
        """
        if value is not None:
            for index, t2 in value.items():
                if t2 < 0:
                    raise Error.ArgumentError("The value of T1 should be a positive real number.")
                if index not in range(self.subSysNum):
                    raise Error.ArgumentError(f"{index}th-qubit is not defined.")
        self._T2 = value

    def diagMat(self) -> ndarray:
        """
        Return the diagonal matrix of the Hamiltonian.

        :return: a diagonal matrix
        """

        levelList = []
        identityList = []

        # Define the list of different identity matrix for each subsystem
        if isinstance(self.sysLevel, int):
            levelList = [self.sysLevel for _ in range(self.subSysNum)]
            identityList = [identity(self.sysLevel) for _ in range(self.subSysNum)]

        elif isinstance(self.sysLevel, list):
            levelList = self.sysLevel
            identityList = [identity(self.sysLevel[i]) for i in range(self.subSysNum)]

        # Initialize the diagonal term
        diagTerm = zeros([self.dim, self.dim], dtype=complex)

        # Add drift terms to the diagonal term
        for index, wq in self.qubitFreq.items():
            numMat = copy.deepcopy(identityList)
            numMat[index] = numMat[index] @ number(levelList[index]).matrix
            diagTerm += wq * tensor(numMat)

        return diagTerm

    def _generateDrift(self, ham: QHamiltonian, setFrame: Dict[int, Union[int, float]] = None) -> None:
        """
        Generate the drift Hamiltonian.

        :param ham: The QHamiltonian object.
        :param setFrame: Hamiltonian in a rotating frame.
        :return: None
        """

        # Define the rotating frame of the system

        if setFrame:
            if isinstance(setFrame, Iterable):
                rotFrame = copy.deepcopy(setFrame)
                for index, freq in self.qubitFreq.items():
                    rotFrq = rotFrame[index]
                    ham.addDrift(operator=number(self.sysLevel), coef=freq - rotFrq, onSubSys=index)

                for index, anharm in self.qubitAnharm.items():
                    ham.addDrift(operator=duff(self.sysLevel), coef=0.5 * anharm, onSubSys=index)
        else:
            for index, anharm in self.qubitAnharm.items():
                ham.addDrift(operator=duff(self.sysLevel), coef=0.5 * anharm, onSubSys=index)

    def _generateTimeIndepCoupTerm(self, ham: QHamiltonian) -> None:
        r"""
        Generate time-independent coupling terms for ith qubit and jth qubit.
        :math:`H_{coup} = a_{i} a_{j}^\dagger + a_{i}^\dagger a_j`

        :param ham: The QHamiltonian object.
        :return: None
        """
        # Generate the time independent coupling terms
        for index, value in self.couplingMap.items():
            ham.addCoupling(onSubSys=[index[0], index[1]], g=value)

    def _generateCoupTerm(self, ham: QHamiltonian) -> None:
        r"""
        Return the time-dependent coupling terms of ith qubit and jth qubit.

        :math:`H_{coup} = (a_i^\dagger a_j + a_i a_j^\dagger) \cos(\omega_{qi} - \omega_{qj})t
            + (a_i^\dagger a_j - a_i a_j^\dagger) i \sin(\omega_{qi} - \omega_{qj})t`

        :param ham: a QHamiltonian object.
        :return: None
        """

        for index, value in self.couplingMap.items():

            # Calculate the detuning of qubit i and qubit j

            deltaOmega = self.qubitFreq[index[0]] - self.qubitFreq[index[1]]

            ai = QOperator('ai')
            aj = QOperator('aj')
            adagi = QOperator('adagi')
            adagj = QOperator('adagj')

            # Define the operators for qubit i and qubit j

            if isinstance(self.sysLevel, int):
                ai = destroy(self.sysLevel)
                adagi = create(self.sysLevel)
                aj = destroy(self.sysLevel)
                adagj = create(self.sysLevel)

            elif isinstance(self.sysLevel, list):
                ai = destroy(self.sysLevel[index[0]])
                adagi = create(self.sysLevel[index[0]])
                aj = destroy(self.sysLevel[index[1]])
                adagj = create(self.sysLevel[index[1]])

            # Add the time-dependent always-on interaction between qubit i and qubit j

            ham.addWave(operators=[adagi, aj], onSubSys=[index[0], index[1]],
                        waves=square(t0=0, t=0, a=value), omega=deltaOmega, phi=0., tag='coupling',
                        flag=QWAVEFORM_FLAG_ALWAYS_ON | QWAVEFORM_FLAG_DO_NOT_CLEAR | QWAVEFORM_FLAG_DO_NOT_PLOT)

            ham.addWave(operators=[ai, adagj], onSubSys=[index[0], index[1]],
                        waves=square(t0=0, t=0, a=value), omega=deltaOmega, phi=0., tag='coupling',
                        flag=QWAVEFORM_FLAG_ALWAYS_ON | QWAVEFORM_FLAG_DO_NOT_CLEAR | QWAVEFORM_FLAG_DO_NOT_PLOT)

            ham.addWave(operators=[QOperator(name='adagi', matrix=1j * adagi.matrix), aj],
                        onSubSys=[index[0], index[1]], waves=square(t0=0, t=0, a=value),
                        omega=deltaOmega, phi=-pi / 2, tag='coupling',
                        flag=QWAVEFORM_FLAG_ALWAYS_ON | QWAVEFORM_FLAG_DO_NOT_CLEAR | QWAVEFORM_FLAG_DO_NOT_PLOT)

            ham.addWave(operators=[QOperator(name='ai', matrix=-1j * ai.matrix), adagj],
                        onSubSys=[index[0], index[1]], waves=square(t0=0, t=0, a=value),
                        omega=deltaOmega, phi=-pi / 2, tag='coupling',
                        flag=QWAVEFORM_FLAG_ALWAYS_ON | QWAVEFORM_FLAG_DO_NOT_CLEAR | QWAVEFORM_FLAG_DO_NOT_PLOT)

    def _setDecoherence(self, ham: QHamiltonian) -> None:
        r"""
        Define the collapse operators accounts for the decoherence noise
        for solving Lindblad master equation of the open system evolution.

        :math:`C_{\rm relaxation} = \frac{1}{\sqrt{T1}} a`
        :math:`C_{\rm dephasing} = \frac{1}{\sqrt{T2}} a^\dagger a`

        :param ham: The QHamiltonian object.
        :return: None
        """
        levelList = []

        # Compute the list of different level for each qubit
        if isinstance(self.sysLevel, list):
            levelList = self.sysLevel

        elif isinstance(self.sysLevel, int):
            levelList = [self.sysLevel for _ in range(self.subSysNum)]

        if self.T1 is None and self.T2 is None:
            pass

        # Initialize the dict type data of T1 and T2 for each qubit
        else:
            t1List = {}
            t2List = {}

            for index in range(self.subSysNum):
                t1List[index] = None
                t2List[index] = None

            if self.T1 is not None:
                for index, t1 in self.T1.items():
                    t1List[index] = t1

            if self.T2 is not None:
                for index, t2 in self.T2.items():
                    t2List[index] = t2

            # Initialize the dict type data for the collapse operators
            cList = {}

            # Calculate the collapse operators for each qubit based on t1List and t2List

            for index in t1List.keys():
                t1 = t1List[index]
                t2 = t2List[index]

                if t1 is None and t2 is None:
                    continue

                elif t2 is not None:
                    cList[index] = []
                    if t1 is not None:
                        rate1 = 1 / t1
                        mat = a(levelList[index])
                        cList[index].append(sqrt(rate1) * mat)

                        rate2 = 0.5 * ((1 / t2) - (1 / (2 * t1)))
                        mat = 2 * adagger(levelList[index]) @ a(levelList[index])
                        cList[index].append(sqrt(rate2) * mat)
                    else:
                        rate2 = 0.5 * (1 / t2)
                        mat = 2 * adagger(levelList[index]) @ a(levelList[index])
                        cList[index].append(sqrt(rate2) * mat)

                elif t1 is not None:
                    cList[index] = []
                    rate1 = 1 / t1
                    mat = a(levelList[index])
                    cList[index].append(sqrt(rate1) * mat)

            ham.setCollapse(cList)

    def _addCrossTalk(self, job: QJob) -> None:
        r"""
        Add classical crosstalk to a QJob object.

        :param job: a QJob object.
        :return: None
        """

        # Create the cross talk matrix of relative drive strength
        std = self.betaStd  # Standard deviation of the relative drive strength
        betaMat = zeros([self.subSysNum, self.subSysNum])
        jobCopy = copy.deepcopy(job)

        # Add standard deviation to the matrix
        for index, value in self.couplingMap.items():
            if index[0] != index[1]:
                beta = random.normal(scale=std)
                betaMat[index[0]][index[1]] = beta
                betaMat[index[1]][index[0]] = beta

        # Create the cross talk matrix of phase lag

        phiMat = zeros([self.subSysNum, self.subSysNum])

        for index, value in self.couplingMap.items():
            if index[0] != index[1]:
                phi = random.random() * 2 * pi
                phiMat[index[0]][index[1]] = phi
                phiMat[index[1]][index[0]] = phi

        ctrlOperatorsDict = copy.deepcopy(jobCopy.ctrlOperators)

        # Add the classical crosstalk to the system

        for opKey, op in ctrlOperatorsDict.items():
            qubitsNum = self.subSysNum
            i = op.onSubSys
            for j in range(0, qubitsNum):
                if betaMat[i][j] != 0:
                    beta = betaMat[i][j]
                    phaseLag = phiMat[i][j]
                    waves = copy.deepcopy(jobCopy.waves[opKey])
                    if isinstance(waves, QWaveform):
                        waves = [waves]
                    for wave in waves:
                        if wave.phi is None:
                            phi = 0
                        else:
                            phi = copy.deepcopy(wave.phi)

                        if wave.omega is not None:
                            omega = copy.deepcopy(wave.omega)
                        else:
                            omega = 0

                        job.addWave(operators=op, onSubSys=j, waves=wave, strength=beta,
                                    omega=omega, phi=phi + phaseLag, tag='cross-talk', flag=QWAVEFORM_FLAG_DO_NOT_PLOT)

    def _addAmpNoise(self, job: QJob) -> None:
        r"""
        Add amplitude distortion to QJob object.

        :param job: a QJob object.
        :return: None
        """

        jobCopy = copy.deepcopy(job)
        jobCopy.clearWaves(tag='coupling')
        jobCopy.clearWaves(tag='cross-talk')
        jobCopy.clearWaves(tag='idle')
        jobCopy.buildWaveCache()

        for key, seq in jobCopy.waveCache.items():
            newSeq = []

            for amp in seq:
                noise = random.normal(scale=self.ampSigma)
                ampNoise = noise * amp  # Define the distortion to the pulse
                newSeq.append(ampNoise)

            job.addWave(name=key, waves=sequence(0, newSeq), tag='amplitude-noise')

    def createQHamiltonian(self, frame: str = "rot") -> QHamiltonian:

        r"""
        Generate a QHamiltonian object based on the physics model and a QJob object.

        :param frame: the rotating frame we choose. The default setting is rotating frame being in the qubits
            frequencies.

        :return: a QHamiltonian object.
        """

        # Initialize the hamiltonian

        ham = QHamiltonian(subSysNum=self.subSysNum, sysLevel=self.sysLevel, dt=self.dt)

        # Create the system Hamiltonian in the lab frame

        if frame == "lab":

            self._generateDrift(ham)
            if self.couplingMap is not None:
                self._generateTimeIndepCoupTerm(ham)
            self._setDecoherence(ham)

        # Create the system Hamiltonian in the rotating frame of qubit frequency

        elif frame == "rot":

            self._generateDrift(ham)
            self._setDecoherence(ham)

            if self.couplingMap is not None:
                self._generateCoupTerm(ham)

        return ham

    def getSimJob(self, job: QJob, noise: bool = True) -> QJob:
        r"""
        Add pulse-related noise to a QJob object.

        :param job: a QJob object.
        :param noise: Add noise to a job if true.

        :return: a QJob object.
        """

        # Check if the subSysNum and sysLevel agrees with the job

        if job.subSysNum != self.subSysNum:
            raise Error.ArgumentError("The variables 'subSysNum' are not matched.")

        if job.sysLevel != self.sysLevel:
            raise Error.ArgumentError("The variables 'sysLevel' are not matched.")

        simJob = copy.deepcopy(job)

        # Add crosstalk noise and amplitude distortion if noise is true

        if noise:
            if self.ampSigma is not None:
                self._addAmpNoise(simJob)

            if self.betaStd is not None:
                self._addCrossTalk(simJob)

        return simJob

