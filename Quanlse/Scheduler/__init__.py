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

**1. Superconduct** (:doc:`Quanlse.Scheduler.Superconduct`)

  - We provide :doc:`Quanlse.Scheduler.Superconduct.DefaultPulseGenerator` using :doc:`remoteOptimizer`
    to users to generate optimal pulses by accessing the Quanlse Cloud Service.

  - We provide :doc:`Quanlse.Scheduler.Superconduct.RBPulseGenerator` using :doc:`remoteOptimizer` especially
    for the :doc:`Quanlse.Utils.RandomizedBenchmarking` functions.

  - We provide :doc:`Quanlse.Scheduler.Superconduct.DefaultPipeline` to users to support the fundamental scheduling
    strategy.

**2. Ion Trap** (:doc:`Quanlse.Scheduler.Ion`)

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
    :param pulseGenerator: the pulseGenerator object.
    :param subSysNum: size of the subsystem
    :param sysLevel: the energy levels of the system (support different levels for different qubits)
    :param savePulse: save the pulse in cache
    """

    def __init__(self, dt: float = None, ham: QHam = None, generator: SchedulerPulseGenerator = None,
                 pipeline: SchedulerPipeline = None, subSysNum: int = None, sysLevel: int = None,
                 savePulse: bool = True):
        """
        Initialization
        """

        if sysLevel is not None:
            self._sysLevel = sysLevel  # type: Union[int, List[int]]
        else:
            self._sysLevel = 2  # type: Union[int, List[int]]

        if subSysNum is not None:
            self._subSysNum = subSysNum  # type: int
        else:
            self._subSysNum = 1  # type: int

        # The quantum register
        self.Q = QRegPool(self)  # type: QRegPool

        # The list for saving the quantum circuit
        self._circuit = []  # type: List['CircuitLine']

        # The configuration dictionary
        self._conf = {}  # type: Dict[str, Any]

        # Sampling duration of AWG (arbitrary wave generator)
        self._dt = dt  # type: float

        # The vault for store the pulses of the fixed gates
        self._pulseCache = []  # type: List[GatePulsePair]

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
        if pipeline is not None:
            self._pipeline = pipeline  # type: SchedulerPipeline
        else:
            self._pipeline = SchedulerPipeline()  # type: SchedulerPipeline

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
    def pulseCache(self) -> List[GatePulsePair]:
        """
        Pulse cache
        """
        return self._pulseCache

    @property
    def pulseGenerator(self) -> Optional[SchedulerPulseGenerator]:
        """
        Pulse cache
        """
        return self._pulseGenerator

    @pulseCache.setter
    def pulseCache(self, obj: List[GatePulsePair]):
        """
        Pulse cache setter
        """
        self._pulseCache = copy.deepcopy(obj)

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

            if sysLevel is None:
                raise Error.ArgumentError("You must input sysLevel when ham is None!")
            else:
                self._sys = sysLevel
            self._job = QJob(self._subSysNum, self._sysLevel, dt)
        else:
            self._tempCtrlOperators = copy.deepcopy(self._ham.job.ctrlOperators)
            self._ham.job = QJob(self.ham.subSysNum, self.ham.sysLevel, self.ham.dt)
            self._ham.job.ctrlOperators = copy.deepcopy(self._tempCtrlOperators)

    def setGatePulsePair(self, *args) -> None:
        """
        Add a GatePulsePair instance to the pulse cache.

        :return: None
        """
        if isinstance(args, GatePulsePair):
            # User input a GatePulsePair object directly
            foundGate = self.findGatePulsePair(args.cirLine.data)
            if foundGate is not None:
                foundGate.pulses = copy.deepcopy(args.pulses)
            else:
                pair = copy.deepcopy(args)
                self._pulseCache.append(pair)
        elif isinstance(args, tuple):
            # User input a gate and a QJob
            if isinstance(args[0].data, (FixedGateOP, RotationGateOP)) and \
                    (isinstance(args[1], QJob) or args[1] is None):
                foundGate = self.findGatePulsePair(args[0])
                if foundGate is not None:
                    foundGate.pulses = copy.deepcopy(args[1])
                else:
                    pair = GatePulsePair(args[0], args[1])
                    self._pulseCache.append(pair)
            else:
                raise Error.ArgumentError(f"Wrong argument, please input a GatePulsePair "
                                          f"or a pair of FixedGate and QJob objects. But you input "
                                          f"a pair of {type(args[0])} with {type(args[1])}.")

    def findGatePulsePair(self, cirLine: CircuitLine) -> Union[GatePulsePair, None]:
        """
        Find the pulse of a fixed gate in pulse cache.

        :param cirLine: CircuitLine object to be searched in pulse cache
        :return: the returned GatePulsePair
        """
        for pair in self._pulseCache:
            if GatePulsePair(cirLine) == pair:
                return pair
        return None

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

    def plotIon(self) -> None:
        """
        Plot the ion pulses.

        :return: None
        """
        if self._ham is not None:
            self._ham.job.plotIon()
        elif self._job is not None:
            self._job.plotIon()

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
        self._pulseCache = []

    def schedule(self) -> QJob:
        """
        Run the scheduler.

        :return: a scheduled QJob object
        """
        # Step 1: generate pulses for the GatePulsePair instances
        # When self._ham is None, it means that the Scheduler does not need to
        # generate pulses for the CircuitLine instances.
        if self._pulseGenerator is not None:
            if self.savePulse:
                for cirLine in self.circuit:
                    foundPair = self.findGatePulsePair(cirLine)
                    if foundPair is None or foundPair.pulses is None:
                        pulseJob = self._pulseGenerator(cirLine, self)
                        self.setGatePulsePair(cirLine, pulseJob)
        else:
            raise Error.ArgumentError("No pulseGenerator is set, hence cannot generate pulse.")

        # Obtain a copy of job
        _tmpJob = copy.deepcopy(self._job) if self._ham is None else copy.deepcopy(self._ham.job)

        # Step 2: Assemble all the pulses according to self.circuit
        _tmpJob.clearWaves()
        for cirLine in self._circuit:
            if self.savePulse:
                _pulses = self.findGatePulsePair(cirLine).pulses
            else:
                _pulses = self._pulseGenerator(cirLine, self)
            _tmpJob.appendJob(_pulses)

        # Step 3: Run the scheduling process
        _tmpJob = self.pipeline(job=_tmpJob, scheduler=self)
        return _tmpJob
