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
Experiment class for superconducting system
"""

import copy
import pickle
import base64
import operator
from typing import List, Any, Dict, Optional, Set

from Quanlse.QOperation import Error
from Quanlse.Superconduct.Lab.ExperimentTask import ExperimentTask
from Quanlse.Superconduct.Lab.LabSpace import LabSpace
from Quanlse.Superconduct.Lab.Runner import Runner
from Quanlse.Superconduct.Lab.Generator import Generator
from Quanlse.Utils.Functions import subspaceVec
from Quanlse.QWaveform import QResult, QJob


class Experiment(object):
    """
    Define experiment on superconducting platform.

    :param labSpace: the LabSpace instance.
    :param runner: the Runner interface for access the hardware.
    :param generator: the PulseGenerator.
    """
    def __init__(self, labSpace: LabSpace = None, runner: Runner = None, generator: Generator = None):
        """ Initialize the object """
        # Set the LabSpace interface
        self.labSpace = labSpace  # type: Optional[LabSpace]
        # Set the Runner interface
        self.runner = runner  # type: Optional[Runner]
        # Set the pulse generator
        self.generator = generator  # type: Optional[Runner]
        # Task list
        self._tasks = []  # type: List[ExperimentTask]
        self._measure = set()  # type: Set[int]

    def __str__(self) -> str:
        """ Print description of the object. """
        returnStr = ""
        for _id, _task in enumerate(self._tasks):
            _qubits = self.labSpace.getQubitLabel(list(_task.measure))
            returnStr += f"Task {_id} `{_task.title}`:\n"
            returnStr += f"    - On qubits: {','.join(_qubits)}\n"
            returnStr += f"    - Description: {_task.description}\n"
            returnStr += f"    - Layers: {_task.layerLength()}\n"
            returnStr += f"    - Gates: {_task.gateSize()}\n"
        return returnStr

    @property
    def measure(self) -> Set[int]:
        """ Return the measured qubits """
        measuredQubit = set()
        for _task in self._tasks:
            measuredQubit = measuredQubit.union(_task.measure)
        return measuredQubit

    @property
    def tasks(self) -> List[ExperimentTask]:
        """ Return the tasks """
        return self._tasks

    def add(self, task: ExperimentTask):
        """
        Add ExperimentTask instance to this Experiment.

        :param task: ExperimentTask
        """
        # Check conflict of measured qubits
        _conflict = self.measure.intersection(task.measure)
        if len(_conflict) > 0:
            raise Error.ArgumentError(f"Can not add task `{task.title}`, because the measured qubit "
                                      f"{_conflict} is conflicted.")
        # Add task
        self._tasks.append(copy.deepcopy(task))

    def assemble(self, tasks: List[ExperimentTask]):
        """
        Assemble the complete job for current task.

        :param tasks: the list of ExperimentTask instances.
        """
        # Assemble all the tasks
        assembledJob = QJob(subSysNum=self.labSpace.subSysNum, dt=self.labSpace.dt)
        taskTiming = {}  # type: Dict[int, float]
        for _taskId, _taskObj in enumerate(tasks):
            taskTiming[_taskId] = 0.
            _layers = _taskObj.layers
            for _layerId in _layers.keys():
                _gates = _layers[_layerId]
                for _gate in _gates:
                    if _gate.job is None:
                        raise Error.RuntimeError(f"Gate `{_gate.name}` of layer `{_layerId}`, task `{_taskObj.title}` "
                                                 f"has no QJob instance, can not run.")
                    assembledJob.appendJob(_gate.job, t0=taskTiming[_taskId])
                    _maxTime, _ = _gate.job.computeMaxTime()
                    taskTiming[_taskId] += _maxTime
        return assembledJob

    def run(self, generator: Generator = None, runner: Runner = None, labSpace: LabSpace = None) -> Dict[int, QResult]:
        """
        Build parallel tasks and execute.

        :param generator: the pulse generator for Quanlse Scheduler.
        :param runner: the runner interface.
        :param labSpace: the config server instance.
        :return: the list of `QResult` instances
        """

        # Set the generator and runner
        if generator is not None:
            self.generator = generator
        if runner is not None:
            self.runner = runner
        if labSpace is not None:
            self.labSpace = labSpace

        _hasScan = False
        for _task in self._tasks:
            if _task.scanDimCount > 0:
                _hasScan = True

        if _hasScan:
            return self.runScan()
        else:
            return self.runNormal()

    def runNormal(self, generator: Generator = None, runner: Runner = None,
                  labSpace: LabSpace = None) -> Dict[int, Any]:
        """
        Build parallel tasks and execute (for the tasks which has no scanning setting).

        :param generator: the pulse generator for Quanlse Scheduler.
        :param runner: the runner interface.
        :param labSpace: the config server instance.
        :return: the list of `QResult` instances
        """

        # Set the generator and runner
        if generator is not None:
            self.generator = generator
        if runner is not None:
            self.runner = runner
        if labSpace is not None:
            self.labSpace = labSpace

        # Check the functional property
        if self.runner is None:
            raise Error.RuntimeError("No runner is set, please set a runner by `expObj.runner = ...`!")
        if self.labSpace is None:
            raise Error.RuntimeError("No labSpace is set, please set a runner by `expObj.labSpace = ...`!")

        # User generator to fulfill the gates without QJob setting
        if self.generator is not None:
            for _task in self._tasks:
                for _gateId in _task.gates.keys():
                    if _task.gates[_gateId].job is None:
                        _task.gates[_gateId].job = self.generator(_task.gates[_gateId])

        # Result dictionary
        _results = {}
        for _taskId, _taskObj in enumerate(self._tasks):
            _results[_taskId] = QResult(title=_taskObj.title, description=_taskObj.description)

        # Run
        _assembledJob = self.assemble(self._tasks)
        _tmpResult = self.runner.run(_assembledJob, self.measure, None, None)

        # Assemble the result
        for _taskId, _taskObj in enumerate(self._tasks):
            # Obtain the qubit indexes of measurement in the returned population vector
            measureIndexInReturnedPop = []
            for qIdx in _taskObj.measure:
                measureIndexInReturnedPop.append(_tmpResult[0]["measure"].index(qIdx))
            _population, _ = subspaceVec(_tmpResult[0]["population"], len(_tmpResult[0]["measure"]), 2,
                                         measureIndexInReturnedPop)
            # Distribute results to the tasks
            _results[_taskId].append({
                "measure": _taskObj.measure,
                "population": _population
            })

        return _results

    def runScan(self, generator: Generator = None, runner: Runner = None,
                labSpace: LabSpace = None) -> Dict[int, Any]:
        """
        Build parallel tasks and execute.

        :param generator: the pulse generator for Quanlse Scheduler.
        :param runner: the runner interface.
        :param labSpace: the config server instance.
        :return: the list of `QResult` instances
        """
        # Set the generator and runner
        if generator is not None:
            self.generator = generator
        if runner is not None:
            self.runner = runner
        if labSpace is not None:
            self.labSpace = labSpace

        # Check the functional property
        if self.runner is None:
            raise Error.RuntimeError("No runner is set, please set a runner by `expObj.runner = ...`!")
        if self.labSpace is None:
            raise Error.RuntimeError("No labSpace is set, please set a runner by `expObj.labSpace = ...`!")

        # Result dictionary
        _results = {}
        for _taskId, _taskObj in enumerate(self._tasks):
            _results[_taskId] = None

        # Separate the normal and scanning tasks
        _normalTasks = []  # type: List[int]
        _scanTasks = []  # type: List[int]
        _tmpNormalExp = Experiment(self.labSpace, self.runner, self.generator)
        for _taskId, _task in enumerate(self._tasks):
            if _task.scanDimCount == 0:
                _normalTasks.append(_taskId)
                _tmpNormalExp.add(_task)
            else:
                _scanTasks.append(_taskId)

        # First, run all tasks which have scanning setting
        if len(_tmpNormalExp.tasks) > 0:
            _normalResults = _tmpNormalExp.runNormal()
            for _normalId, _originalId in enumerate(_normalTasks):
                _results[_originalId] = _normalResults[_normalId]

        # Generate scanning region list
        _scanRegion = {}
        for _taskId in _scanTasks:
            _taskObj = self._tasks[_taskId]
            _scanRegion[_taskId] = {}
            for _dim, _setting in _taskObj.scanSetting.items():
                _scanRegion[_taskId][_dim] = _setting['scan_setting'].getList()

        def _taskStepper():
            """
            Assemble all the pulse parameters
            """
            # Initialize the parameters
            _regions = {}
            _paramsLenCheck = {}
            _paramsLen = {}
            _paramsReg = {}
            _paramsCnt = {}
            for __taskId in _scanTasks:
                __taskObj = self._tasks[__taskId]
                _regions[__taskId] = [_item["scan_setting"].steps for _item in __taskObj.scanSetting.values()]
                _paramsLenCheck[__taskId] = [_region - 1 for _region in _regions[__taskId]]
                _paramsLenCheck[__taskId][-1] += 1
                _paramsLen[__taskId] = [_region for _region in _regions[__taskId]]
                _paramsReg[__taskId] = [0 for _ in range(__taskObj.scanDimCount)]
                _paramsCnt[__taskId] = __taskObj.scanDimCount

            roundCount = 0
            _currentParamsReg = {}
            while True:
                _currentParamsReg = {}
                # Initial the assembled parameters
                _assembledParameters = {}  # The key is parameter key
                _assembledMeasureIdx = set([])
                # For each task
                roundCount += 1
                _currentTaskIds = []
                for __taskId in _scanTasks:
                    __taskObj = self._tasks[__taskId]
                    # Insert the scan parameters into `_assembledParameters` if the traverse is not finished yet.
                    if not operator.le(_paramsLenCheck[__taskId], _paramsReg[__taskId]):
                        _currentTaskIds.append(__taskId)
                        # Assemble the parameters
                        _assembledParameters[__taskId] = {}
                        for __dim, __setting in __taskObj.scanSetting.items():
                            _assembledParameters[__taskId][_dim] = {
                                "layer": _setting["layer"], "gate_id": _setting["gate_id"],
                                "wave_key": _setting["wave_key"], "wave_id": _setting["wave_id"],
                                "arg_key": _setting["scan_arg_key"] if "scan_arg_key" in _setting else None,
                                "setting_key": _setting["scan_setting_key"] if "scan_setting_key" in _setting else None
                            }
                        # If contains parameters, then assemble the measured qubit indexes
                        for _qIdx in __taskObj.measure:
                            _assembledMeasureIdx.add(_qIdx)
                        # Save the current reg
                        _currentParamsReg[__taskId] = copy.deepcopy(_paramsReg[__taskId])
                        # Counting: the last region add 1
                        _paramsReg[__taskId][_paramsCnt[__taskId] - 1] += 1
                        # Calculate the indexes and carry
                        for _addParamIdx in reversed(range(_paramsCnt[__taskId])):
                            if _paramsReg[__taskId][_addParamIdx] == _paramsLen[__taskId][_addParamIdx]:
                                # Carry
                                if _addParamIdx <= 0:
                                    break
                                else:
                                    _paramsReg[__taskId][_addParamIdx - 1] += 1
                                    _paramsReg[__taskId][_addParamIdx] = 0
                            else:
                                # Do not need to carry
                                break
                if len(_currentTaskIds) > 0:
                    yield {
                        "task_id_list": _currentTaskIds,
                        "measured_qubits": _assembledMeasureIdx,
                        "region_id": _currentParamsReg
                    }
                else:
                    yield None

        _stepper = _taskStepper()
        _round = next(_stepper)
        step = 0

        while _round is not None:
            _taskIds = _round["task_id_list"]
            step += 1
            # Assemble the tasks into one NormalTask
            _tmpScanExp = Experiment(labSpace=self.labSpace, runner=self.runner, generator=self.generator)
            for _taskId in _taskIds:
                if _taskId not in _round["task_id_list"]:
                    continue
                _taskObj = self._tasks[_taskId]
                _tmpTask = ExperimentTask(measure=_taskObj.measure, title=f"step {step} in {_taskObj.title}",
                                          description=_taskObj.description)
                # Map task dim to gate dim
                _taskDimMapToGateDim = {}
                for _dimId, _setting in _taskObj.scanSetting.items():
                    if _setting['gate_id'] not in _taskDimMapToGateDim.keys():
                        _taskDimMapToGateDim[_setting['gate_id']] = []
                    _taskDimMapToGateDim[_setting['gate_id']].append(_dimId)
                # Traverse the layers
                for _layerId, _gateIds in _taskObj.layerIds.items():
                    for _gateId in _gateIds:
                        _gateObj = _taskObj.gates[_gateId]
                        _gateSubDim = _gateObj.scanDim()
                        if _gateSubDim <= 0:
                            # If the current gate has no scanning setting
                            _tmpTask.addControlGate(ctrlGate=_gateObj, layer=_layerId, onQubit=_gateObj.onQubit)
                            continue
                        # If _scanDim > 0
                        # print(f"    gateSubDim={_gateSubDim}, _taskId={_taskId}, gate_id={_gateId}")
                        _gateJobDict = _gateObj.job.dump2Dict()
                        for _currentGateDim in range(_gateSubDim):
                            if _round["region_id"][_taskId] is None:
                                continue
                            _taskDim = _taskDimMapToGateDim[_gateId][_currentGateDim]
                            # If the current gate has scanning setting
                            _regionId = _round["region_id"][_taskId][_taskDim]
                            _value = _scanRegion[_taskId][_taskDim][_regionId]
                            _channel = _taskObj.scanSetting[_taskDim]['wave_key']
                            _layer = _taskObj.scanSetting[_taskDim]['layer']
                            _waveId = _taskObj.scanSetting[_taskDim]['wave_id']
                            # Set fixed parameter
                            if "scan_arg_key" in _taskObj.scanSetting[_taskDim]:
                                _paraKey = _taskObj.scanSetting[_taskDim]["scan_arg_key"]
                                _gateJobDict["waves"][_channel][_waveId]["args"][_paraKey] = _value
                            elif "scan_setting_key" in _taskObj.scanSetting[_taskDim]:
                                _paraKey = _taskObj.scanSetting[_taskDim]["scan_setting_key"]
                                _gateJobDict["waves"][_channel][_waveId][_paraKey] = _value
                        _newGateObj = copy.deepcopy(_gateObj)
                        _newGateObj.job = QJob.parseJson(_gateJobDict)
                        _tmpTask.addControlGate(ctrlGate=_newGateObj, layer=_layerId, onQubit=_newGateObj.onQubit)
                _tmpScanExp.add(_tmpTask)
            # Run the experiment
            _taskResults = _tmpScanExp.runNormal()
            for _subTaskId, _res in _taskResults.items():
                _originalTaskId = _taskIds[_subTaskId]
                if _originalTaskId not in _results.keys() or _results[_originalTaskId] is None:
                    _results[_originalTaskId] = QResult()
                _results[_originalTaskId].result.append(_res[0])
            # Get the next step
            _round = next(_stepper)
        return _results

    def clone(self) -> 'Experiment':
        """
        Return the copy of the object
        """
        return copy.deepcopy(self)

    def dump(self) -> str:
        """
        Return a base64 encoded string.

        :return: encoded string
        """
        obj = copy.deepcopy(self)
        # Verify
        if obj.runner is not None:
            raise Error.RuntimeError("The runner is set, hence the instance can not be serialized.")
        if obj.generator is not None:
            raise Error.RuntimeError("The generator is set, hence the instance can not be serialized.")
        # Remove the config server
        obj._configServer = None
        # Serialization
        byteStr = pickle.dumps(obj)
        base64str = base64.b64encode(byteStr)
        del obj
        return base64str.decode()

    @staticmethod
    def load(base64Str: str) -> 'Experiment':
        """
        Create object from base64 encoded string.

        :param base64Str: a base64 encoded string
        :return: ExperimentScan object
        """
        byteStr = base64.b64decode(base64Str.encode())
        obj = pickle.loads(byteStr)  # type: Experiment

        return obj
