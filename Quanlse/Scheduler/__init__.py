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
Quanlse's Scheduler is a module that allows for automatic generation of fidelity-optimized and
scheduled pulse sequence for a given quantum circuit set to perform a certain quantum computing task.

Quanlse's Scheduler has the following benefits:

- Highly automatic: it generates high-fidelity pulses automatically and simultaneously
  minimizes the overall gate operation time.

- Practical: it considers several limitations of the superconducting quantum system,
  including leakage errors, decoherence, etc.

- Flexible: it gives users the freedom to customize qubits and circuit parameters.
  This can also be easily extended to other quantum computing platforms.

At present, Quanlse Scheduler support two different quantum platforms:

**1. Superconduct** (:doc:`Quanlse.Superconduct.SchedulerSupport`)

  - We provide :doc:`Quanlse.Superconduct.SchedulerSupport.GeneratorCloud` using :doc:`remoteOptimizer`
    to users to generate optimal pulses by accessing the Quanlse Cloud Service.

  - We provide :doc:`Quanlse.Superconduct.SchedulerSupport.RBPulseGenerator` using :doc:`remoteOptimizer` especially
    for the :doc:`Quanlse.Utils.RandomizedBenchmarking` functions.

  - We provide :doc:`Quanlse.Superconduct.SchedulerSupport.PipelineCenterAligned` to users to support the fundamental
    scheduling strategy.

**2. Ion Trap** (:doc:`Quanlse.TrappedIon.SchedulerSupport`)

For more details, please visit https://quanlse.baidu.com/#/doc/tutorial-scheduler.
"""

import copy
from typing import List, Union, Optional, Dict, Any

import numpy

from Quanlse.QHamiltonian import QHamiltonian as QHam
from Quanlse.QOperation.FixedGate import FixedGateOP
from Quanlse.QOperation.RotationGate import RotationGateOP
from Quanlse.QWaveform import QJob
from Quanlse.QRegPool import QRegPool
from Quanlse.QOperation import Error, CircuitLine
from Quanlse.Scheduler.SchedulerPipeline import SchedulerPipeline
from Quanlse.Scheduler.SchedulerPulseGenerator import SchedulerPulseGenerator
from Quanlse.Scheduler.GatePulsePair import GatePulsePair


class Scheduler:
    """
    Basic class of Quanlse Scheduler.

    :param dt: AWG sampling time
    :param ham: the QHamiltonian object.
    :param generator: the pulseGenerator object.
    :param subSysNum: size of the subsystem
    :param sysLevel: the energy levels of the system (support different levels for different qubits)
    :param savePulse: save the pulse in cache
    """

    def __init__(self, dt: float = None, ham: QHam = None, generator: SchedulerPulseGenerator = None,
                 pipeline: SchedulerPipeline = None, subSysNum: int = None,
                 sysLevel: Union[int, List[int], None] = None, savePulse: bool = True):
        """
        Initialization
        """

        self._sysLevel = sysLevel  # type: Union[int, List[int], None]

        if subSysNum is not None:
            self._subSysNum = subSysNum  # type: Optional[int]
        else:
            self._subSysNum = None  # type: Optional[int]

        # The quantum register
        self.Q = QRegPool(self)  # type: QRegPool

        # The list for saving the quantum circuit
        self._circuit = []  # type: List['CircuitLine']

        # The configuration dictionary
        self._conf = {}  # type: Dict[str, Any]

        # Sampling duration of AWG (arbitrary wave generator)
        self._dt = dt  # type: float

        # The vault for store the pulses of the fixed gates
        self._gatePulsePairs = []  # type: List[GatePulsePair]

        # The time padding of the pulses of adjacent quantum gates
        self.padding = 0.0  # type: float

        # Save the control operators
        self._tempCtrlOperators = {}  # type: Dict[str, Any]

        # The QHam object, this is not necessary
        self._ham = None  # type: Optional[QHam]
        if ham is not None:
            self._ham = copy.deepcopy(ham)
            self._ham.dt = dt
            self._subSysNum = ham.subSysNum
            self._sysLevel = ham.sysLevel

        # Add the pulse generator
        if generator is not None:
            self._pulseGenerator = generator  # type: Optional[SchedulerPulseGenerator]
        else:
            self._pulseGenerator = None  # type: Optional[SchedulerPulseGenerator]

        # If save the pulse for quantum gates in cache
        self._savePulse = savePulse  # type: bool

        # Add the pipeline
        self._pipeline = None  # type: Optional[SchedulerPipeline]
        if pipeline is not None:
            self.pipeline = pipeline  # type: SchedulerPipeline
        else:
            self.pipeline = SchedulerPipeline()  # type: SchedulerPipeline

        # The QJob object, this is not necessary
        self._initializeQJob(dt, subSysNum, sysLevel)

    @property
    def conf(self) -> Dict[str, Any]:
        """
        Scheduler Configuration
        """
        return self._conf

    @property
    def pipeline(self) -> SchedulerPipeline:
        """
        Scheduler Pipeline
        """
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: SchedulerPipeline):
        """
        Pipeline setter
        """
        if self._pipeline is not None:
            del self._pipeline
        self._pipeline = pipeline
        self._pipeline.scheduler = self

    @property
    def savePulse(self) -> bool:
        """
        If save the pulse for quantum gates in cache
        """
        return self._savePulse

    @savePulse.setter
    def savePulse(self, value: bool):
        """
        If save the pulse for quantum gates in cache
        """
        self._savePulse = value

    @property
    def subSysNum(self) -> int:
        """
        Return the number of sub-systems.
        """
        return self._subSysNum

    @property
    def sysLevel(self) -> Union[int, List[int]]:
        """
        Return the size of the sub-systems.
        """
        return self._sysLevel

    @property
    def dt(self) -> float:
        """
        AWG Sampling time
        """
        return self._dt

    @property
    def circuit(self) -> List[CircuitLine]:
        """
        Circuit information
        """
        return self._circuit

    @circuit.setter
    def circuit(self, circuit: List[CircuitLine]):
        """
        Circuit setter
        """
        self._circuit = circuit
        self._initializeQJob(self._dt, self._subSysNum, self._sysLevel)

    @property
    def pulseGenerator(self) -> Optional[SchedulerPulseGenerator]:
        """
        Pulse cache
        """
        return self._pulseGenerator

    @property
    def gatePulsePairs(self) -> List[GatePulsePair]:
        """
        Pulse cache
        """
        return self._gatePulsePairs

    @gatePulsePairs.setter
    def gatePulsePairs(self, obj: List[GatePulsePair]):
        """
        Pulse cache setter
        """
        self._gatePulsePairs = copy.deepcopy(obj)

    @property
    def ham(self) -> Optional[QHam]:
        """
        Respective Hamiltonian
        """
        if self._ham is None:
            return None
        else:
            return self._ham

    @ham.setter
    def ham(self, obj: QHam):
        """
        Hamiltonian setter
        """
        self._ham = copy.deepcopy(obj)
        self._subSysNum = obj.subSysNum
        self._sysLevel = obj.sysLevel

    @property
    def job(self) -> QJob:
        """
        QJob in Scheduler
        """
        return self._job

    def _initializeQJob(self, dt, subSysNum: int = None, sysLevel: int = None) -> None:
        """
        Initialize a QJob object.

        :param dt: AWG sampling time
        :param subSysNum: sybsystem's number
        :param sysLevel: system's level
        :return: None
        """
        if self._ham is None:
            if subSysNum is None:
                for cirLin in self._circuit:
                    if self._subSysNum < cirLin.qRegIndexList.max() + 1:
                        self._subSysNum = cirLin.qRegIndexList.max() + 1
            else:
                self._subSysNum = subSysNum

            self._sys = sysLevel
            self._job = QJob(self._subSysNum, self._sysLevel, dt)
        else:
            self._tempCtrlOperators = copy.deepcopy(self._ham.job.ctrlOperators)
            self._ham.job = QJob(self.ham.subSysNum, self.ham.sysLevel, self.ham.dt)
            self._ham.job.ctrlOperators = copy.deepcopy(self._tempCtrlOperators)

    def appendGatePulsePair(self, pair: GatePulsePair):
        """
        Append a GatePulsePair to scheduler.

        :param pair: the GatePulsePair instance
        """
        if self.subSysNum is not None:
            if isinstance(pair.qubits, int):
                if pair.qubits >= self.subSysNum:
                    raise Error.ArgumentError(f"The index of pair's qubits ({pair.qubits}) exceeds the scheduler's "
                                              f"subSysNum ({self.subSysNum}).")
            else:
                if max(pair.qubits) >= self.subSysNum:
                    raise Error.ArgumentError(f"The index of pair's qubits ({pair.qubits}) exceeds the scheduler's "
                                              f"subSysNum ({self.subSysNum}).")
        _pair = copy.deepcopy(pair)
        if _pair.job is None:
            if self._pulseGenerator is not None:
                cirLine = CircuitLine(_pair.gate, [_pair.qubits] if isinstance(_pair.qubits, int) else _pair.qubits)
                job = self._pulseGenerator(cirLine, self)
                _pair.job = job
            else:
                _pair.job = None

        self.gatePulsePairs.append(_pair)

    def addGatePulsePair(self, gate: Union[FixedGateOP, RotationGateOP], qubits: Union[int, List[int]],
                         t0: float = None, job: QJob = None) -> None:
        """
        Add GatePulsePair to scheduler.

        :param gate: the FixedGateOP or RotationGateOP object.
        :param qubits: indicates the subsystem of the gate performing on
        :param job: the QJob object
        :param t0: the start time
        """
        if self.subSysNum is not None:
            if isinstance(qubits, int):
                if qubits >= self.subSysNum:
                    raise Error.ArgumentError(f"The index of qubits ({qubits}) exceeds the scheduler's "
                                              f"subSysNum ({self.subSysNum}).")
            else:
                if max(qubits) >= self.subSysNum:
                    raise Error.ArgumentError(f"The index of qubits ({qubits}) exceeds the scheduler's "
                                              f"subSysNum ({self.subSysNum}).")
        if len(self._circuit) > 0:
            raise Error.ArgumentError("The logical circuit is not None, please clear the circuit and then add " 
                                      "the gate pulse pair.")
        self._addGatePulsePair(gate, qubits, t0, job)

    def _addGatePulsePair(self, gate: Union[FixedGateOP, RotationGateOP], qubits: Union[int, List[int]],
                          t0: float = None, job: QJob = None) -> None:
        """
        Add GatePulsePair to scheduler.

        :param gate: the FixedGateOP or RotationGateOP object.
        :param qubits: indicates the subsystem of the gate performing on
        :param job: the QJob object
        :param t0: the start time
        """
        if job is None:
            if self._pulseGenerator is not None:
                cirLine = CircuitLine(gate, [qubits] if isinstance(qubits, int) else qubits)
                job = self._pulseGenerator(cirLine, self)
            else:
                job = None
        # Add the gate pulse pair into the list
        pair = GatePulsePair(gate, qubits, t0, job)
        self._gatePulsePairs.append(pair)

    def getMatrix(self) -> numpy.ndarray:
        """
        Return the full matrix of circuit.
        """
        dim = 2 ** self.subSysNum
        # Initialize the matrix
        cirMat = numpy.identity(dim)
        # Traverse the circuitLine
        for cir in self.circuit:
            # Obtain basic data
            _mat = cir.data.getMatrix()
            _onQubits = cir.qRegIndexList
            if max(_onQubits) >= self.subSysNum:
                raise Error.ArgumentError("Qubit index(es) exceeds the scheduler's subSysNum.")
            if 1 <= len(_onQubits) <= 2:
                if _mat.shape[0] != 2 ** len(_onQubits):
                    raise Error.ArgumentError(f"The matrix of {len(_onQubits)}-qubit gate must be {2 ** len(_onQubits)}"
                                              f"-dimensional, however its dimension is {_mat.shape[0]}.")
            else:
                raise Error.ArgumentError("Only single- or two-qubit gates are supported!")
            if len(_onQubits) > 1 and abs(_onQubits[0] - _onQubits[1]) > 1:
                raise Error.ArgumentError("The indexes of two qubit gates must be close.")
            # Generate the matrix
            _preIdm = numpy.identity(2 ** min(_onQubits))
            _postIdm = numpy.identity(2 ** (self.subSysNum - max(_onQubits) - 1))
            cirMat = numpy.kron(numpy.kron(_preIdm, _mat), _postIdm) @ cirMat

        return cirMat

    def plot(self) -> None:
        """
        Plot the pulses.

        :return: None
        """
        if self._ham is not None:
            self._ham.job.plot()
        elif self._job is not None:
            self._job.plot()

    def clearCircuit(self) -> None:
        """
        Clear all circuit.

        :return: None
        """
        self._circuit = []
        if self._ham is not None:
            self._ham.job.clearWaves()
        elif self._job is not None:
            self._job.clearWaves()

    def clearCache(self) -> None:
        """
        Clear all cache.

        :return: None
        """
        self.clearGatePulsePair()

    def clearGatePulsePair(self) -> None:
        """
        Clear all gatePulsePairs.

        :return: None
        """
        self._gatePulsePairs = []

    def schedule(self) -> QJob:
        """
        Run the scheduler.

        :return: a scheduled QJob object
        """
        # Step 1: Generate pulses for the GatePulsePair instances
        # When self._ham is None, it means that the Scheduler does not need to
        # generate pulses for the CircuitLine instances.
        if self._pulseGenerator is not None:
            _lastGatePulsePairDuration = 0.
            self.clearGatePulsePair()
            for cirLine in self.circuit:
                self._addGatePulsePair(cirLine.data, cirLine.qRegIndexList, _lastGatePulsePairDuration)
                _lastGatePulsePairDuration += self.gatePulsePairs[-1].t
        else:
            raise Error.ArgumentError("No pulseGenerator is set, hence cannot generate pulse.")

        # Step 2: Run the scheduling process
        self.pipeline()

        # Step 3: Transform the GatePulsePair list into a QJob instance
        _resultJob = copy.deepcopy(self._job) if self._ham is None else copy.deepcopy(self._ham.job)
        _resultJob.clearWaves()
        for _pair in self.gatePulsePairs:
            _t0 = _pair.t0
            # Add waves
            for _waveKeys in _pair.job.waves.keys():
                for _wave in _pair.job.waves[_waveKeys]:
                    _wave.t0 += _t0
                    if _waveKeys not in _resultJob.waves.keys():
                        _resultJob.waves[_waveKeys] = []
                    _resultJob.waves[_waveKeys].append(_wave)
            # Add other data
            for _opKey in _pair.job.ctrlOperators.keys():
                _resultJob.ctrlOperators[_opKey] = _pair.job.ctrlOperators[_opKey]
            for _opKey in _pair.job.LO.keys():
                _resultJob.LO[_opKey] = _pair.job.LO[_opKey]

        return _resultJob
