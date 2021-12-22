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
ExperimentTask class for superconducting system
"""

from typing import List, Union, Dict, Set, Any

from Quanlse.QOperation import Error
from Quanlse.QWaveform import QJob
from Quanlse.Utils.ControlGate import ControlGate
from Quanlse.Superconduct.Lab.Utils import Scan


class ExperimentTask(object):
    """
    Define an Experiment task. A task is the minimal unit for defining the task, multi ExperimentTask can be running
    in parallel in Experiment object.

    :param measure: the list of measured qubits in this task
    :param title: the title of the task
    :param description: the description of the task
    """
    def __init__(self, measure: Union[int, Set[int]], title: str = "", description: str = ""):
        """ Initialize the object """
        # Basic information
        self.title = title  # type: str
        self.description = description  # type: str
        # Save the GatePulsePair Id list of each layer
        self._layers = {}  # type: Dict[int, List[int]]
        # Save all the control gates.
        self._gateCount = 0
        self._gates = {}  # type: Dict[int, ControlGate]
        self._measure = set([measure] if isinstance(measure, int) else measure)  # type: Set[int]
        # Scan parameters
        self._scanDimCount = 0  # type: int
        self._scanSetting = {}  # type: Dict[int, Dict[str, Any]]

    def __len__(self):
        """ Return the length of layers """
        return self.layerLength()

    def __sizeof__(self):
        """ Return the size of gates """
        return self.gateSize()

    def __str__(self) -> str:
        """ Print description of the object. """
        returnStr = ""
        for _layerId in self._layers:
            returnStr += f"Layer {_layerId}:\n"
            for _gateId in self._layers[_layerId]:
                _gate = self._gates[_gateId]
                returnStr += f"    - Gate {_gate.name} on qubit {_gate.onQubit}: "
                if _gate.job is None:
                    returnStr += f"no QJob is set.\n"
                else:
                    _waveKeys = list(_gate.job.waves.keys())
                    returnStr += f"contains {len(_waveKeys)} waveforms on {','.join(_waveKeys)}.\n"
        return returnStr

    @property
    def measure(self) -> Set[int]:
        """ Return the measured qubits """
        return self._measure

    @measure.setter
    def measure(self, measure: Union[int, List[int]]):
        """ Return the measured qubits """
        self._measure = measure

    @property
    def gates(self) -> Dict[int, ControlGate]:
        """ Return the gate list """
        return self._gates

    @property
    def layers(self) -> Dict[int, List[ControlGate]]:
        """ Return the gate list of every layer """
        layerList = {}  # type: Dict[int, List[ControlGate]]
        for _layerId in self._layers.keys():
            layerList[_layerId] = []
            for _gateId in self._layers[_layerId]:
                layerList[_layerId].append(self._gates[_gateId])
        return layerList

    @property
    def layerIds(self) -> Dict[int, List[int]]:
        """ Return the gate ID list  of every layer"""
        return self._layers

    @property
    def scanDimCount(self) -> int:
        """ The dimension of scanning """
        return self._scanDimCount

    @property
    def scanSetting(self) -> Dict[int, Dict[str, Any]]:
        """ The detail setting of scanning """
        return self._scanSetting

    def layerLength(self) -> int:
        """ Return the length of layers """
        return len(self._layers)

    def gateSize(self) -> int:
        """ Return the size of gates """
        return len(self._gates.keys())

    def _searchScanningSetting(self):
        """
        Search scanning settings.
        """
        self._scanDimCount = 0
        for _layer in self._layers.keys():
            _gateIds = self._layers[_layer]
            for _gateId in _gateIds:
                _gate = self._gates[_gateId]  # type: ControlGate
                if _gate.job is None:
                    continue
                for _waveKey in _gate.job.waves.keys():
                    for _waveId, _wave in enumerate(_gate.job.waves[_waveKey]):
                        _waveDict = _wave.dump2Dict()
                        for _settingKey in _waveDict.keys():
                            if isinstance(_waveDict[_settingKey], Scan):
                                self._scanSetting[self._scanDimCount] = {
                                    "layer": _layer, "gate_id": _gateId, "wave_key": _waveKey, "wave_id": _waveId,
                                    "scan_setting_key": _settingKey, "scan_setting": _waveDict[_settingKey]
                                }
                                self._scanDimCount += 1
                        for _argKey in _waveDict['args'].keys():
                            if isinstance(_waveDict['args'][_argKey], Scan):
                                self._scanSetting[self._scanDimCount] = {
                                    "layer": _layer, "gate_id": _gateId, "wave_key": _waveKey, "wave_id": _waveId,
                                    "scan_arg_key": _argKey, "scan_setting": _waveDict['args'][_argKey]
                                }
                                self._scanDimCount += 1

    def addControlGate(self, ctrlGate: ControlGate, layer: int, onQubit: Union[List[int], int] = None) -> int:
        """
        Add a ControlGate into the task.

        :param ctrlGate: the instance of ControlGate
        :param layer: the layer of the ctrlGate added to
        :param onQubit: the list of qubits of the ctrlGate added to
        """
        if self.layerLength() < layer:
            raise Error.ArgumentError(f"Current experiment task has {self.layerLength()} layers, hence you can not "
                                      f"add gate to the layer number {layer}, the maximum available layer number is "
                                      f"{self.layerLength()}.")
        # Check conflict
        if self.layerLength() > layer:
            _allOnQubits = set()
            for _gateId in self._layers[layer]:
                _gate = self._gates[_gateId]  # type: ControlGate
                _allOnQubits.union(set([_gate.onQubit] if isinstance(_gate.onQubit, int) else _gate.onQubit))
            _intersection = _allOnQubits & set([onQubit] if isinstance(onQubit, int) else onQubit)
            if len(_intersection) > 0:
                raise Error.ArgumentError(f"Can not add new gate `{ctrlGate.name}`, because qubit "
                                          f"{list(_intersection)} is already occupied by another gate.")

        # Add gate
        _currentId = self._gateCount
        self._gateCount += 1
        self._gates[_currentId] = ctrlGate

        # Add to layer
        if layer not in self._layers:
            self._layers[layer] = []
        self._layers[layer].append(_currentId)

        # Add scanning setting
        self._searchScanningSetting()

        return _currentId

    def addGate(self, name: str, layer: int, onQubit: Union[List[int], int], job: QJob = None):
        """
        Add gate to experiment task.

        :param name: the name of the ControlGate
        :param layer: the layer of the ctrlGate added to
        :param onQubit: the list of qubits of the ctrlGate added to
        :param job: the control pulse (QJob instance) of the ControlGate
        """
        # Create a ControlGate instance.
        ctrlGate = ControlGate(name, onQubit, job)
        _currentId = self.addControlGate(ctrlGate, layer, onQubit)

    def appendGate(self, name: str, onQubit: Union[List[int], int], job: QJob = None):
        """
        Append gate to the end of the experiment task.

        :param name: the name of the ControlGate
        :param onQubit: the list of qubits of the ctrlGate added to
        :param job: the control pulse (QJob instance) of the ControlGate
        """
        # Create a ControlGate instance.
        ctrlGate = ControlGate(name, onQubit, job)
        _currentId = self.addControlGate(ctrlGate, self.layerLength(), onQubit)
