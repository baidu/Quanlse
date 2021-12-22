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
The utils for Quanlse Simulator
"""
import copy
from numpy import real, square
from typing import Dict, List, Any, Union, Set

from Quanlse.QWaveform import QJob, QResult
from Quanlse.QOperator import QOperator
from Quanlse.Utils.Functions import basis, project, subspaceVec
from Quanlse.QPlatform import Error
from Quanlse.Superconduct.Lab import LabSpace, Runner, Generator
from Quanlse.Superconduct.Simulator.PulseModel import PulseModel
from Quanlse.Utils.ODESolver import solverAdaptive as solveHam


def _solveHam(ham: 'QHamiltonian', state0=None, shot=None, recordEvolution=False, accelerate=None):
    """ Run Hamiltonian simulation """
    _res = solveHam(ham, state0, shot, recordEvolution, accelerate)
    _qRes = QResult()
    _qRes.append(_res)
    return _qRes


class SimulatorLabSpace(LabSpace):
    """
    LabSpace for Quanlse Simulator.
    """

    def __init__(self, configDict: Dict[str, Any]):
        """ Initialization """
        if configDict is None:
            self._configDict = {}
        else:
            if not isinstance(configDict, dict):
                raise Error.ArgumentError("configDict must be a dictionary.")
            self._configDict = copy.deepcopy(configDict)
        super().__init__()

    def _addConfigToServer(self, keys: Union[str, List[str]], values: List[Any]) -> bool:
        """ Add config to configDict """

        # Create configItemList
        if isinstance(keys, str):
            keys = [keys]
        if isinstance(values, str):
            values = [values]

        for _idx, _key in enumerate(keys):
            _cateList = _key.split('.')
            self._configDict[_key] = values[_idx]
        return True

    def _readFromServer(self, keys: List[str], timestamp: int = None, ignoreCache: bool = False) -> Any:
        """ Read config from the dict. """
        if keys is None or len(keys) < 1:
            return self._configDict
        else:
            returned = {}
            for _key in keys:
                if _key in self._configDict.keys():
                    returned[_key] = self._configDict[_key]
                else:
                    returned[_key] = None
            return returned

    def _obtainLabSpaceList(self, timeRange: List[int] = None) -> List[Any]:
        raise Error.RuntimeError("Can not require Lab Space list in local simulator model.")

    def _createLabSpaceToServer(self, labSpaceID: str, templateLabSpaceID: str = None, description: str = None) -> Dict:
        raise Error.RuntimeError("Can not create Lab Space in local simulator model.")

    def _setChannels(self, channels: Dict[str, Any]):
        """
        Set the channel settings.
        """
        for _qubit in channels.keys():
            if _qubit in self._QubitMapLabel2Id.keys():
                self._channels[_qubit] = copy.deepcopy(channels[_qubit])

    def _setQubits(self, qubits: List[str]):
        """
        Set the qubit names.
        """
        for _id, _name in enumerate(qubits):
            if _name in self._QubitMapLabel2Id.keys():
                raise Error.ArgumentError(f"Qubit name is repeated: {_name}.")
            self._QubitMapLabel2Id[_name] = _id
            self._QubitMapId2Label[_id] = _name


class SimulatorRunner(Runner):
    """ The experiment runner for pulseSim """

    def __init__(self, model: PulseModel = None):
        """ Initialization """
        super().__init__()
        self.model = model  # type: PulseModel
        self.channelMapping = {}  # type: Dict[str, QOperator]

    def replaceSymbolOperator(self, job: QJob):
        """ Replace the symbol operator with the solid operator """
        _job = copy.deepcopy(job)
        for _opKey in _job.ctrlOperators:
            _opList = _job.ctrlOperators[_opKey]
            if isinstance(_opList, QOperator):
                _opList = [_opList]
            for _op in _opList:
                if _op.name not in self.channelMapping.keys():
                    raise Error.ArgumentError(f"The solid operator for symbol operator {_op.name} is not defined.")
                _solidOp = self.channelMapping[_op.name]
                _op.name = _solidOp.name
                _op.coef = _solidOp.coef
                _op.matrix = copy.deepcopy(_solidOp.matrix)

        return _job

    def setChannelOperator(self, channelName: str, operator: QOperator):
        """
        Set the solid control operator (the QOperator instance which contains non-empty matrix property) for channels.

        :param channelName: the channel name
        :param operator: solid operator
        """
        if operator.matrix is None:
            raise Error.ArgumentError("The input operator contains no matrix.")
        self.channelMapping[channelName] = copy.deepcopy(operator)

    def run(self, job: QJob, measure: Set[int], scheduler: 'Scheduler' = None, conf: Dict[Any, Any] = None) -> QResult:
        """
        Run the simulator
        """
        if self.model is None:
            raise Error.ArgumentError("Model is not set in this runner!")
        # Replace the symbol operators with the solid operators
        solidJob = self.replaceSymbolOperator(job)
        # Run the simulation
        _ham = copy.deepcopy(self.model.ham)
        _ham.job = solidJob
        result = _solveHam(ham=_ham, state0=basis(self.model.dim, 0))
        # Project the final state to computational space
        state = real(project(square(abs(result[0]["state"])).T[0], self.model.subSysNum, self.model.sysLevel, 2))
        # Trace out the qubits which are not in the measured qubit indexes
        _pop, _basis = subspaceVec(state, self.model.subSysNum, 2, list(measure))
        result[0]["population"] = list(_pop)
        result[0]["measure"] = list(measure)
        return result


class SimulatorGenerator(Generator):
    """ The experiment runner for pulseSim """

    def __init__(self, labSpace: LabSpace):
        """ Initialization """
        super().__init__(labSpace=labSpace)
