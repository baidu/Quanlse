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
Quantum Environment
"""
import importlib
import json
from copy import copy
from typing import List, Dict, TYPE_CHECKING, Union, Tuple, Optional

import numpy

from Quanlse.Define import sdkVersion, noLocalTask, circuitPackageFile
from Quanlse.QPlatform.Utilities import numpyMatrixToDictMatrix, dictMatrixToNumpyMatrix
from Quanlse.QRegPool import QRegPool
from Quanlse import BackendName, Algorithm, HardwareImplementation, GateTimeDict, TStepRange
from Quanlse.QOperation.RotationGate import RotationGateOP
from Quanlse.QTask import _uploadCircuit, _createTask, _waitTask
from Quanlse.QPlatform import Error

if TYPE_CHECKING:
    from Quanlse.QOperation import CircuitLine
    from Quanlse.Pulse import QPulseImplement

class QuanlseEnv:
    """
    Quantum Environment class
    """

    def __init__(self) -> None:
        self.Q = QRegPool(self)

        self.gateTimeDict = copy(GateTimeDict)

        self.circuit = []  # type: List['CircuitLine']

        self.backendName = None  # type: Optional['BackendName']

        self.qRegCount = 0
        self.params = []  # type: List[Tuple[int, float, float]]
        self.coupling = None  # type: Optional[numpy.ndarray]
        self.opt_a_bound = {}  # type: Dict[str, Tuple[float, float]]

        self.algorithm = None  # type: Optional[Algorithm]
        self.hardwareImplementation = None  # type: Optional[HardwareImplementation]
        self.tStep = 0.

        self.program = None  # type: Optional[Dict]

    def backend(self, backendName: 'BackendName', algorithm: Algorithm, hardwareImplementation: HardwareImplementation,
                tStep: float) -> None:
        if type(backendName) is not str:
            backendName = backendName.value
        self.backendName = backendName

        self.algorithm = algorithm

        if hardwareImplementation != HardwareImplementation.CZ:
            raise Error.ArgumentError(f'UnImplemented Quanlse hardware {hardwareImplementation}')
        self.hardwareImplementation = hardwareImplementation

        if tStep < TStepRange[0] or tStep > TStepRange[1]:
            raise Error.ArgumentError(f'tStep({tStep}) out of range')
        self.tStep = tStep

    def setupQReg(self, qRegSettingList: List[Tuple[float, float]]):
        self.qRegCount = len(qRegSettingList)
        self.coupling = numpy.zeros((self.qRegCount, self.qRegCount))
        for setting in qRegSettingList:
            frequency = setting[0]
            anharmonicity = setting[1]
            self.params.append(
                (len(self.params), frequency, anharmonicity)  # Tuple
            )

    def setupCoupling(self, couplingSettingList: List[Tuple[Tuple[int, int], float, Optional[Tuple[float, float]]]]):
        for setting in couplingSettingList:
            qRegPartner = setting[0]
            strength = setting[1]
            a_bound = setting[2]
            self.coupling[qRegPartner[0]][qRegPartner[1]] = strength
            self.coupling[qRegPartner[1]][qRegPartner[0]] = strength
            self.opt_a_bound[f'{qRegPartner[0]}-{qRegPartner[1]}'] = a_bound

    def publish(self) -> None:
        circuit = []
        for circuitLine in self.circuit:
            if isinstance(circuitLine.data, RotationGateOP):
                argumentList = circuitLine.data.argumentList
            else:
                argumentList = None
            circuit.append({
                'gate': circuitLine.data.name,
                'argumentList': argumentList,
                'qRegIndexList': circuitLine.qRegIndexList,
                'gateTime': circuitLine.gateTime,
            })

        self.program = {
            'sdkVersion': sdkVersion,
            'Head': {
                'qRegCount': self.qRegCount,
                'params': self.params,
                'coupling': numpyMatrixToDictMatrix(self.coupling),
                'opt_a_bound': self.opt_a_bound,
            },
            'Body': {
                'Circuit': circuit,
            },
        }

    def commit(self, downloadResult=True, debug: str = "") -> Dict[str, Union[str, Dict[str, int]]]:
        """
        Switch local/cloud commitment by prefix of backend name
        """

        self.publish()

        if self.backendName.startswith('local_'):
            return self._localCommit()
        elif self.backendName.startswith('cloud_'):
            return self._cloudCommit(downloadResult, debug)
        else:
            raise Error.ArgumentError(f"Invalid backendName => {self.backendName}")

    def _localCommit(self) -> Dict[str, Union[str, Dict[str, int]]]:
        """
        Local commitment

        :return: task result
        """

        if noLocalTask is not None:
            raise Error.RuntimeError('Local tasks are not allowed in the online environment!')

        # import the backend plugin according to the backend name
        module = _loadPythonModule(
            f'Quanlse.Pulse.{self.backendName}')
        if module is None:
            raise Error.ArgumentError(f"Invalid local backend => {self.backendName}!")
        backendClass = getattr(module, 'Backend')

        # configure the parameters
        backend = backendClass()  # type: 'QuantumPulseImplement'
        backend.program = self.program
        backend.algorithm = self.algorithm
        backend.hardwareImplementation = self.hardwareImplementation
        backend.tStep = self.tStep
        # execution
        return backend.commit()

    def _cloudCommit(self, downloadResult: bool, debug: str) -> Dict[str, Union[str, Dict[str, int]]]:
        """
        Cloud Commitment

        :return: task result
        """

        circuitId = None
        taskId = None
        self.publish()  # circuit in Protobuf format
        programBuf = json.dumps(self.program)
        with open(circuitPackageFile, 'wb') as file:
            file.write(programBuf.encode("utf-8"))

        # todo process the file and upload failed case
        token, circuitId = _uploadCircuit(circuitPackageFile)
        backend = self.backendName[6:]  # omit the prefix `cloud_`
        taskId = _createTask(token, circuitId, self.algorithm,
                             self.hardwareImplementation, self.tStep, debug)

        outputInfo = True
        if outputInfo:
            print(f"Circuit upload successful, circuitId => {circuitId} taskId => {taskId}")

        taskResult = _waitTask(token, taskId, downloadResult=downloadResult)
        if type(taskResult) == str:
            raise Error.RuntimeError(taskResult)

        return taskResult


def _loadPythonModule(moduleName: str):
    """
    Load module from file system.

    :param moduleName: module name
    :return: module object
    """

    moduleSpec = importlib.util.find_spec(moduleName)
    if moduleSpec is None:
        return None
    module = importlib.util.module_from_spec(moduleSpec)
    moduleSpec.loader.exec_module(module)
    return module
