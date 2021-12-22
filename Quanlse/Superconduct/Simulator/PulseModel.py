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
import numpy
from typing import List, Union, Optional, Dict, Tuple, Callable, Any
from numpy import pi, sqrt, ndarray, zeros, identity, random, cos, sin, trace, round

from Quanlse.QHamiltonian import QHamiltonian
from Quanlse.QPlatform import Error
from Quanlse.QOperator import QOperator, duff, create, destroy, a, adagger, number, driveX, uWave
from Quanlse.QWaveform import QJob, square, QWaveform, QJobList, QResult, sequence, \
    QWAVEFORM_FLAG_ALWAYS_ON, QWAVEFORM_FLAG_DO_NOT_CLEAR, QWAVEFORM_FLAG_DO_NOT_PLOT
from Quanlse.Utils.Functions import tensor, basis, dagger
from Quanlse.Superconduct.SchedulerSupport import SchedulerSuperconduct, SchedulerPulseGenerator
from Quanlse.QPlatform.Error import ArgumentError


class PulseModel(SchedulerSuperconduct):
    """
    Basic class of PulseModel. It defines the system information and noise-related parameters. It is also a sub-class of
    SchedulerSuperconduct.

    :param subSysNum: size of the subsystem
    :param sysLevel: the energy levels of the system (support different levels for different qubits)
    :param qubitFreq: qubit frequencies, in 2 pi GHz
    :param driveFreq: drive frequencies, in 2 pi GHz
    :param qubitAnharm: Anharmonicity for each qubit
    :param dt: sampling time period:
    :param couplingMap: direct coupling structure
    :param T1: T1-relaxation time, in nanoseconds
    :param T2: T2-dephasing time, in nanoseconds
    :param ampSigma: deviation of the amplitude distortion
    :param betaStd: standard deviation of the relative strength in classical crosstalk
    :param frameMode: rotating frame (in qubit frequency)
    :param pulseGenerator: customized pulse generator for scheduler
    """

    def __init__(self,
                 subSysNum: int,
                 sysLevel: Union[int, List[int]],
                 qubitFreq: Dict[int, Union[int, float]],
                 qubitAnharm: Dict[int, Union[int, float]],
                 dt: float = 0.2,
                 driveFreq: Dict[int, Union[int, float]] = None,
                 couplingMap: Optional[Dict[Tuple, Union[int, float]]] = None,
                 T1: Optional[Dict[int, Union[int, float, None]]] = None,
                 T2: Optional[Dict[int, Union[int, float, None]]] = None,
                 ampSigma: Optional[Union[int, float]] = None,
                 betaStd: Optional[Union[int, float]] = None,
                 frameMode: str = 'rot',
                 pulseGenerator: Callable = None):

        self.frameMode = frameMode  # type: str

        self._subSysNum = subSysNum  # type: int
        self._sysLevel = sysLevel  # type: int
        self._dt = dt  # type: float
        self._conf = {}  # type: Dict

        self._qubitFreq = qubitFreq  # type: Dict[int, Union[int, float]]
        self._driveFreq = {} if driveFreq is None else driveFreq  # type: Dict[int, Union[int, float]]
        self._qubitAnharm = qubitAnharm  # type: Dict[int, Union[int, float]]
        self._T1 = T1  # type: Optional[Dict[int, Union[int, float]]]
        self._T2 = T2  # type: Optional[Dict[int, Union[int, float]]]
        self._couplingMap = couplingMap  # type: Optional[Dict[Tuple, Union[int, float]]]
        self._pulseGenerator = None  # type: Optional[SchedulerPulseGenerator]
        self._pulseGeneratorFunc = pulseGenerator  # type: Callable

        super(PulseModel, self).__init__(dt, ham=None, subSysNum=subSysNum,
                                         sysLevel=sysLevel, generator=None)

        self.createQHamiltonian(frameMode=self.frameMode)

        self.betaStd = betaStd  # type: Optional[Union[int, float]]
        self.ampSigma = ampSigma  # type: Optional[Union[int, float]]

        if self._pulseGeneratorFunc is not None:
            self._pulseGenerator = self._pulseGeneratorFunc(self.ham)

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
        self.createQHamiltonian(frameMode=self.frameMode)

    @property
    def driveFreq(self) -> Dict[int, Union[int, float]]:
        """
        Return the drive frequency of each qubit in the system.
        """
        return self._qubitFreq

    @driveFreq.setter
    def driveFreq(self, value: Dict[int, Union[int, float]]):
        """
        Return the drive frequency of each qubit in the system, in 2 * pi * GHz.
        """
        if len(value) != self.subSysNum:
            raise Error.ArgumentError("The number of the frequencies and the number of qubits are not the same.")

        self._driveFreq = value
        self.createQHamiltonian(frameMode=self.frameMode)

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
        self.createQHamiltonian(frameMode=self.frameMode)

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
                    raise Error.ArgumentError("The number of the frequencies and the number "
                                              "of qubits are not the same.")

        self._couplingMap = value
        self.createQHamiltonian(frameMode=self.frameMode)

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
        self.createQHamiltonian(frameMode=self.frameMode)

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
        self.createQHamiltonian(frameMode=self.frameMode)

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

    def _generateDrift(self, ham: QHamiltonian, frameMode: str = 'rot') -> None:
        """
        Generate the drift Hamiltonian.

        :param ham: The QHamiltonian object.
        :param frameMode: Hamiltonian in a rotating frame.
        :return: None
        """

        # Define the rotating frame of the system

        if frameMode == 'lab':

            # create drift term in the lab frame

            if isinstance(self.sysLevel, int):
                for index, freq in self.qubitFreq.items():
                    ham.addDrift(operator=number(self.sysLevel), coef=freq, onSubSys=index)
                for index, anharm in self.qubitAnharm.items():
                    ham.addDrift(operator=duff(self.sysLevel), coef=0.5 * anharm, onSubSys=index)
            elif isinstance(self.sysLevel, list):
                for index, freq in self.qubitFreq.items():
                    ham.addDrift(operator=number(self.sysLevel[index]), coef=freq, onSubSys=index)
                for index, anharm in self.qubitAnharm.items():
                    ham.addDrift(operator=duff(self.sysLevel[index]), coef=0.5 * anharm, onSubSys=index)

        elif frameMode == 'rot':

            # create drift term in the rotating frame

            if isinstance(self.sysLevel, int):

                for index, anharm in self.qubitAnharm.items():
                    ham.addDrift(operator=duff(self.sysLevel), coef=0.5 * anharm, onSubSys=index)

            if isinstance(self.sysLevel, list):
                for index, anharm in self.qubitAnharm.items():
                    ham.addDrift(operator=duff(self.sysLevel[index]), coef=0.5 * anharm, onSubSys=index)

        else:
            raise Error.ArgumentError(f"Unsupported frameMode {frameMode}, only 'lab' and 'rot' are supported.")

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
                        waves=square(t=0, a=value), freq=deltaOmega, phase=0., tag='coupling', t0=0,
                        flag=QWAVEFORM_FLAG_ALWAYS_ON | QWAVEFORM_FLAG_DO_NOT_CLEAR | QWAVEFORM_FLAG_DO_NOT_PLOT)

            ham.addWave(operators=[ai, adagj], onSubSys=[index[0], index[1]],
                        waves=square(t=0, a=value), freq=deltaOmega, phase=0., tag='coupling', t0=0,
                        flag=QWAVEFORM_FLAG_ALWAYS_ON | QWAVEFORM_FLAG_DO_NOT_CLEAR | QWAVEFORM_FLAG_DO_NOT_PLOT)

            ham.addWave(operators=[QOperator(name='adagi', matrix=1j * adagi.matrix), aj],
                        onSubSys=[index[0], index[1]], waves=square(t=0, a=value),
                        freq=deltaOmega, phase=-pi / 2, tag='coupling', t0=0,
                        flag=QWAVEFORM_FLAG_ALWAYS_ON | QWAVEFORM_FLAG_DO_NOT_CLEAR | QWAVEFORM_FLAG_DO_NOT_PLOT)

            ham.addWave(operators=[QOperator(name='ai', matrix=-1j * ai.matrix), adagj],
                        onSubSys=[index[0], index[1]], waves=square(t=0, a=value),
                        freq=deltaOmega, phase=-pi / 2, tag='coupling', t0=0,
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
                        if wave.phase is None:
                            phase = 0
                        else:
                            phase = copy.deepcopy(wave.phase)

                        if wave.freq is not None:
                            freq = copy.deepcopy(wave.freq)
                        else:
                            freq = 0

                        job.addWave(operators=op, onSubSys=j, waves=wave, strength=beta,
                                    freq=freq, phase=phase + phaseLag, tag='cross-talk',
                                    flag=QWAVEFORM_FLAG_DO_NOT_PLOT)

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

            job.addWave(name=key, waves=sequence(newSeq), t0=0, tag='amplitude-noise')

    def createQHamiltonian(self, frameMode: str = "rot") -> QHamiltonian:
        r"""
        Generate a QHamiltonian object based on the physics model and a QJob object.

        :param frameMode: the rotating frame we choose. The default setting is rotating frame being in the qubits
            frequencies.
        :return: a QHamiltonian object.
        """

        # Initialize the hamiltonian
        ham = QHamiltonian(subSysNum=self.subSysNum, sysLevel=self.sysLevel, dt=self.dt)

        # Create the system Hamiltonian in the lab frame
        if frameMode == "lab":
            # Create the system Hamiltonian in the lab frame
            self._generateDrift(ham, frameMode='lab')
            if self.couplingMap is not None:
                self._generateTimeIndepCoupTerm(ham)
            self._setDecoherence(ham)
        elif frameMode == "rot":
            # Create the system Hamiltonian in the rotating frame of qubit frequency
            self._generateDrift(ham)
            self._setDecoherence(ham)
            if self.couplingMap is not None:
                self._generateCoupTerm(ham)
        else:
            raise Error.ArgumentError(f"Unsupported frameMode {frameMode}, only 'lab' and 'rot' are supported.")

        self.ham = ham

        if self._driveFreq is not None:
            # Set the local oscillator
            for qIdx in self._driveFreq.keys():
                self.ham.job.setLO(uWave, qIdx, freq=self._driveFreq[qIdx])

        if self._pulseGeneratorFunc is not None:
            self._pulseGenerator = self._pulseGeneratorFunc(self.ham)

        return self.ham

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

        if job.sysLevel is not None and job.sysLevel != self.sysLevel:
            raise Error.ArgumentError("The variables 'sysLevel' are not matched.")

        simJob = copy.deepcopy(job)

        # Add crosstalk noise and amplitude distortion if noise is true

        if noise:
            if self.ampSigma is not None:
                self._addAmpNoise(simJob)

            if self.betaStd is not None:
                self._addCrossTalk(simJob)

        return simJob

    def simulate(self, job: QJob = None, state0: ndarray = None, jobList: QJobList = None,
                 measure: List[int] = None, shot: int = None, options: Any = None) -> QResult:
        """
        Calculate the unitary evolution operator with a given Hamiltonian. This function supports
        both single-job and batch-job processing.

        :param job: the QJob object to simulate.
        :param state0: the initial state vector. If None is given, this function will return the time-ordered
                       evolution operator, otherwise returns the final state vector.
        :param jobList: a job list containing the waveform list
        :param measure: indicates the qubit indexes for measurement
        :param shot: return the population of the eigenstates when ``shot`` is provided
        :param options: the options for simulation
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

        if self.frameMode == 'lab':
            if state0 is None:
                state0 = basis(self.dim, 0)
            if shot is None:
                shot = 1000

        _useHamSimulator = True

        # Simulate the QJobs
        if _useHamSimulator:
            result = self._ham.simulate(job=job, state0=state0, jobList=jobList, shot=shot, adaptive=True)
        else:
            raise Error.RuntimeError('Not implemented')

        # Measure the population
        if measure is not None:
            if state0 is not None:
                pass
            else:
                raise ArgumentError("When 'measure' is given, you must input 'state0' or 'shot'!")

        return result


class ReadoutPulse:
    """
    A object contains the readout pulses to probe the resonator for qubit readout and demodulation.

    :param driveStrength: drive strength of the readout pulse.
    :param driveFreq: drive frequency of the readout pulse.
    :param loFreq: local oscillator's frequency for demodulation.
    """

    def __init__(self, driveStrength: Dict[int, Any], driveFreq: Dict[int, Any], loFreq: Union[int, float]):

        self._driveStrength = driveStrength
        self._driveFreq = driveFreq
        self._loFreq = loFreq

    @property
    def driveStrength(self) -> Dict[int, Any]:
        """
        Drive strength of readout pulse.
        """
        return self._driveStrength

    @driveStrength.setter
    def driveStrength(self, value: Dict[int, Any]):
        """
        Define the drive strength of readout pulse.
        """
        self._driveStrength = value

    @property
    def driveFreq(self) -> Dict[int, Any]:
        """
        Return the readout frequencies.
        """
        return self._driveFreq

    @driveFreq.setter
    def driveFreq(self, value: Dict[int, float]):
        """
        Define the drive frequencies of readout pulse.
        """
        self._driveFreq = value

    @property
    def loFreq(self):
        """
        Carrier frequency generated by the local oscillator of signal demodulation.
        """
        return self._loFreq

    @loFreq.setter
    def loFreq(self, value: float):
        """
        Set the carrier frequency generated by the local oscillator for signal demodulation.
        """
        self._loFreq = value

    def setFreq(self, index: int, freq: Union[int, float]):
        """
        Set the measurement channel's drive frequency of specified qubit.
        """
        self.driveFreq[index] = freq

    def setStrength(self, index: int, strength: Union[int, float]):
        """
        Set the strength of the measurement channel of specified qubit.
        """

        self.driveStrength[index] = strength


class ReadoutModel:
    """
    Basic class of readout simulation model, contains hardware information of the readout cavity.

    :param pulseModel: physics model of the simulator.
    :param resonatorFreq: bare frequencies of the resonators.
    :param level: energy levels of the resonators.
    :param coupling: qubit-resonator coupling strength.
    :param dissipation: dissipation parameter of the resonator.
    :param gain: the gain of the amplification.
    :param dt: sampling period of the AWG.
    :param impedance: impedance of the transmission line.
    :param conversionLoss: the conversion loss of the transmission line.
    """

    def __init__(self,
                 pulseModel: PulseModel,
                 resonatorFreq: Dict[int, float],
                 level: int,
                 coupling: Dict[int, float],
                 dissipation: Union[int, float],
                 gain: Union[int, float],
                 dt: float = 1.,
                 impedance: float = 1.,
                 conversionLoss: float = 1.):
        """
        The constructor of the readout model.
        """

        self.pulseModel = pulseModel  # type: PulseModel
        self.resonatorFreq = resonatorFreq  # type: Dict[int, float]
        self.dissipation = dissipation  # type:  Union[int, float]
        self.gain = gain  # type: Union[int, float]
        self.level = level  # type: int
        self.coupling = coupling  # type: Dict[int, float]
        self.dt = dt  # type: float
        self.impedance = impedance  # type: float
        self.conversionLoss = conversionLoss  # type: float

        self._readoutPulse = None  # type: Optional[ReadoutPulse]

    @property
    def resonatorFreq(self) -> Dict[int, float]:
        """
        Return the resonator frequencies.
        """

        return self._resonatorFreq

    @resonatorFreq.setter
    def resonatorFreq(self, value: Dict[int, float]):
        """
        Set the resonator frequencies.
        """
        qubitIdx = self.pulseModel.qubitFreq.keys()
        for resonatorIdx in value.keys():
            if resonatorIdx not in qubitIdx:
                raise Error.ArgumentError("Resonator indexes not found in qubit indexes")

            self._resonatorFreq = value

    @property
    def readoutPulse(self) -> ReadoutPulse:
        """
        The multiplexing pulse data for qubit readout.
        """
        return self._readoutPulse

    @readoutPulse.setter
    def readoutPulse(self, value: ReadoutPulse):
        """
        Set the measure channel for the readout
        """

        self._readoutPulse = value

    def _createHam(self, idx: int, driveFreq: Union[int, float]) -> QHamiltonian:
        """
        Return the hamiltonian of qubit-resonator circuit model according to the qubit index and drive frequency.

        :param idx: index of the qubit.
        :param driveFreq: drive frequency.

        :return: QHamiltonian in the rotating frame.
        """
        qubitFreq = self.pulseModel.qubitFreq
        qubitAnharm = self.pulseModel.qubitAnharm
        resonatorFreq = self.resonatorFreq
        qubitLevel = self.pulseModel.sysLevel
        cList = {1: [sqrt(self.dissipation) * a(self.level)]}

        ham = QHamiltonian(subSysNum=2, sysLevel=[qubitLevel, self.level], dt=self.dt)
        ham.addDrift(operator=number(qubitLevel), onSubSys=0, coef=qubitFreq[idx] - driveFreq)
        ham.addDrift(operator=duff(qubitLevel), onSubSys=0, coef=0.5 * qubitAnharm[idx])
        ham.addDrift(operator=number(self.level), onSubSys=1, coef=resonatorFreq[idx] - driveFreq)
        ham.addCoupling(onSubSys=[0, 1], g=self.coupling[idx])
        ham.setCollapse(cList)

        return ham

    def _createHamLab(self, idx: int) -> QHamiltonian:
        """
        Return the hamiltonian of qubit-resonator circuit model according to the qubit index and drive frequency.

        :param idx: index of the qubit.

        :return: QHamiltonian in the lab frame.
        """

        # Extract information from PulseModel object
        qubitFreq = self.pulseModel.qubitFreq
        qubitAnharm = self.pulseModel.qubitAnharm
        resonatorFreq = self.resonatorFreq
        qubitLevel = self.pulseModel.sysLevel
        cList = {1: [sqrt(self.dissipation) * a(self.level)]}

        # Create lab frame QHamiltonian object of a coupled transmon-resonator system
        ham = QHamiltonian(subSysNum=2, sysLevel=[qubitLevel, self.level], dt=self.dt)
        ham.addDrift(operator=number(qubitLevel), onSubSys=0, coef=qubitFreq[idx])
        ham.addDrift(operator=duff(qubitLevel), onSubSys=0, coef=0.5 * qubitAnharm[idx])
        ham.addDrift(operator=number(self.level), onSubSys=1, coef=resonatorFreq[idx])
        ham.addCoupling(onSubSys=[0, 1], g=self.coupling[idx])
        ham.setCollapse(cList)

        return ham

    def _readoutJob(self, amp: Union[int, float], duration: Union[int, float]) -> QJob:
        """
        Return readout pulse in rotating frame at driving frequency.

        :param amp: amplitude of the readout job.
        :param duration: the duration of the pulse.

        :return: QJob.
        """
        qubitLevel = self.pulseModel.sysLevel
        job = QJob(subSysNum=2, sysLevel=[qubitLevel, self.level], dt=self.dt)
        wave = square(t=duration, a=amp)
        job.addWave(operators=driveX, onSubSys=1, waves=wave, t0=0.)
        return job

    def simulate(self, duration: Union[int, float], resIdx: Optional[List[int]] = None,
                 state: 'str' = 'ground') -> Dict:
        """
        Simulate the state evolution of the qubit-resonator system using Jaynes-Cumming model.

        :param duration: pulse duration of the measurement.
        :param resIdx: indexes of the resonator the pulse acted on.
        :param state: qubit state prepared in ground state or in excited state, Optional parameter 'ground' or 'excited'

        :return: None.
        """

        if self._readoutPulse is None:
            raise Error.ArgumentError("You need to define object of class: ReadoutPulse")

        # Extract information for qubit readout simulation
        k = self.conversionLoss
        kappa = self.dissipation
        g = self.gain
        z = self.impedance
        qubitLevel = self.pulseModel.sysLevel

        # check if the qubit index exists
        if resIdx is not None:
            for idx in resIdx:
                if idx not in range(self.pulseModel.subSysNum):
                    raise Error.ArgumentError(f"Qubit index {idx} not found")
        else:
            resIdx = range(self.pulseModel.subSysNum)

        readoutPulse = self.readoutPulse

        if state == 'ground':
            psi = tensor(basis(qubitLevel, 0), basis(self.level, 0))
        elif state == 'excited':
            psi = tensor(basis(qubitLevel, 1), basis(self.level, 0))
        else:
            raise Error.ArgumentError("The input qubit state is 'ground' or 'excited'")

        # set the frequency of local oscillator for demodulation
        omegaLo = readoutPulse.loFreq

        # initialize a dict object to record the evolution
        rhoHistory = {}

        # define annihilation and creation operators
        a = tensor([identity(qubitLevel), destroy(self.level).matrix])
        adag = dagger(a)

        # define quadrature operators
        x = 0.5 * (adag + a)
        p = 1j * 0.5 * (adag - a)

        viRawList = {}
        vqRawList = {}
        viList = {}
        vqList = {}
        rhoList = {}

        # simulate the readout dynamic of the designated qubits
        for idx in resIdx:
            # run simulation of the given qubit
            ham = self._createHam(idx=idx, driveFreq=readoutPulse.driveFreq[idx])
            job = self._readoutJob(amp=readoutPulse.driveStrength[idx], duration=duration)
            res0 = ham.simulate(state0=psi, job=job, isOpen=True, recordEvolution=True)

            # Append the evolution of density matrix to idx-th qubit
            rhoHistory[idx] = res0[0]['evolution_history']
            _, maxDt = job.computeMaxTime()
            timeIdx = range(0, int(maxDt))

            # Calculate the intermediate frequency
            omegaRf = self.readoutPulse.driveFreq[idx]
            omegaIf = abs(omegaLo - omegaRf)
            viRawList[idx] = []
            vqRawList[idx] = []
            vif = k * readoutPulse.driveStrength[idx] * sqrt(kappa * g * z * omegaRf / 2)

            # Record the raw data of vi, vq quadratures for each dt
            for dtIdx in timeIdx:

                nowTime = dtIdx * self.dt
                nowRho = rhoHistory[idx][dtIdx]

                vi = vif * (x * cos(omegaIf * nowTime) - p * sin(omegaIf * nowTime))

                vq = - vif * (p * cos(omegaIf * nowTime) + x * sin(omegaIf * nowTime))

                viRawList[idx].append(trace(vi @ nowRho).real)
                vqRawList[idx].append(trace(vq @ nowRho).real)

            viList[idx] = trace(vif * x @ rhoHistory[idx][-1]).real
            vqList[idx] = trace(-vif * p @ rhoHistory[idx][-1]).real

            rhoList[idx] = rhoHistory[idx][-1]

        # Return the result in the form of dictionary data type
        readoutData = {'vi': viList,
                       'vq': vqList,
                       'viRaw': viList,
                       'vqRaw': vqList,
                       'index': resIdx,
                       'duration': duration,
                       'dt': self.dt,
                       'loFreq': omegaLo,
                       'rho': rhoList}

        return readoutData
