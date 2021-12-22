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
LabSpace for superconducting system experiments
"""
import copy
import time
from typing import Dict, Any, Union, List, Optional

from Quanlse.QPlatform import Error


class LabSpace(object):
    """
    The configuration server for pulse parameters. Users can overwrite this class to implement the data service.

    :param labSpaceID: the labSpace ID.
    :param conf: the user-defined options/configurations.
    """

    def __init__(self, labSpaceID: str = None, conf: Dict[Any, Any] = None):
        """ Initialize the DataServer """
        self._conf = conf
        # The local setting of configuration, has a higher priority than the server configuration.
        self._localConfig = {}  # type: Dict[Any, Any]
        # Default time stamp for obtain the config, when it is None, ConfigServer will return the latest config.
        self._defaultTimeStamp = None  # type: Union[int, None]
        # LabSpace ID
        self._labSpaceID = labSpaceID  # type: str
        # Define the mapping between qubit label and Index
        self._QubitMapLabel2Id = {}  # type: Dict[str, int]
        self._QubitMapId2Label = {}  # type: Dict[int, str]
        # Define the control channels of the hardware
        self._channels = {}  # type: Dict[str, Any]
        # Basic information
        self.device = None  # type: Optional[str]
        self.subSysNum = None  # type: Optional[int]
        self.dt = None  # type: Optional[float]
        self.title = None  # type: Optional[str]
        self.description = None  # type: Optional[str]
        # Load from LabSpace server
        _basicConfig = ["basic.device", "basic.awg_dt", "basic.sub_system_number", "basic.title",
                        "basic.description", "basic.qubits", "basic.channels"]
        _config = self.getConfig(_basicConfig)
        if _config is not None:
            # Check
            for _key in _basicConfig:
                if _key not in _config.keys():
                    raise Error.RuntimeError(f"Can not read config item `{_key}`!")
            # Set properties
            self.device = _config["basic.device"]
            self.dt = _config["basic.awg_dt"]
            self.subSysNum = _config["basic.sub_system_number"]
            self.title = _config["basic.title"]
            self.description = _config["basic.description"]
            self._setQubits(list(_config["basic.qubits"].values()))
            self._setChannels(_config["basic.channels"])
        else:
            raise Error.RuntimeError("Can not obtain config from LabSpace.")

    def __getitem__(self, item: str) -> Any:
        """ Get certain task """
        return self.getConfig(item)

    @property
    def localConfig(self) -> Dict[Any, Any]:
        """
        Get the reference of ExperimentRunner.
        """
        return self._localConfig

    @property
    def conf(self) -> Dict[Any, Any]:
        """
        Get the reference of ExperimentRunner.
        """
        return self._conf

    @conf.setter
    def conf(self, value: Dict[Any, Any]):
        """
        Set the ConfigServer instance.
        """
        self._conf = value

    @property
    def labSpaceID(self) -> str:
        """
        Get the labSpaceID.
        """
        return self._labSpaceID

    @labSpaceID.setter
    def labSpaceID(self, value: str):
        """
        Set the labSpaceID.
        """
        self._labSpaceID = value

    def _createLabSpaceToServer(self, labSpaceID: str, templateLabSpaceID: str = None, description: str = None) -> Dict:
        """
        Create a new labSpace to server.

        :param labSpaceID: the labSpace ID of the new labSpace
        :param templateLabSpaceID: the new labSpace will copy from an existing labSpace if this parameter is given
        :param description: the description of the labSpace
        :return: information about the new labSpace
        """
        raise NotImplementedError("Abstract method `_createLabSpaceToServer()` is not implemented!")

    def _obtainLabSpaceList(self, timeRange: List[int] = None) -> List[Any]:
        """
        Obtain the full labSpace list on server.

        :param timeRange: the time range list, the format is `[start_timestamp, end_timestamp]`.
        :return: the full list of the labSpaces
        """
        raise NotImplementedError("Abstract method `_obtainLabSpaceList()` is not implemented!")

    def _readFromServer(self, keys: List[str], timestamp: int = None, ignoreCache: bool = False) -> Dict[str, Any]:
        """
        Read from server with cate list.

        Notice: `keys` may be a key or a list of keys.

        :param keys: a list which containing the key hierarchy of specific config
        :param timestamp: the version of config at specific timestamp, pass None to obtain the latest version
        :param ignoreCache: ignore the local cache
        :return: the value of the config
        """
        raise NotImplementedError("Abstract method `_readFromServer()` is not implemented!")

    def _addConfigToServer(self, keys: Union[str, List[str]], values: List[Any]) -> bool:
        """
        Add a config to ConfigServer.

        :param keys: the key of the config
        :param values: the value of the config
        :return: True if successfully added
        """
        raise NotImplementedError("Abstract method `addConfig()` is not implemented!")

    def setLocalConfig(self, config: Dict[Any, Any]):
        """
        Set local config which has a higher priority than the server configuration.

        :param config: dictionary type object
        """
        if config is not None:
            for _key in config.keys():
                self.localConfig[_key] = config[_key]

    def removeLocalConfig(self, keys: List[str]):
        """
        remove local config.

        :keys: the key of local config
        """
        if keys is not None:
            for _key in keys:
                self.localConfig.pop(_key)

    def clearLocalConfig(self):
        """
        Clear all local config
        """
        self.localConfig.clear()

    def addDict(self, configDict: Dict[str, Any]) -> bool:
        """
        Add a config to ConfigServer by Python Dictionary.

        :param configDict: the python dictionary formatted configs
        :return: True if successfully added
        """
        keys = []
        values = []

        # Check input parameters
        def _traverse(_rootKey: List[str], _dict: Dict):
            """ Traverse all the keys. """
            for _key in _dict.keys():
                if isinstance(_dict[_key], dict):
                    _currentKey = copy.copy(_rootKey)
                    _currentKey.append(_key)
                    _traverse(_currentKey, _dict[_key])
                else:
                    valueKey = copy.copy(_rootKey)
                    valueKey.append(_key)
                    keys.append('.'.join(valueKey))
                    values.append(_dict[_key])
        _traverse([], configDict)
        # Then update the remote config
        return self._addConfigToServer(keys, values)

    def addConfig(self, keys: Union[str, List[str]], values: Union[Any, List[Any]]) -> bool:
        """
        Add a config to ConfigServer.

        :param keys: the keys of the config
        :param values: the values of the config
        :return: True if successfully added
        """
        # Check input parameters
        if keys is None or len(keys) < 1:
            raise Error.RuntimeError("Please input valid key(s) by pass parameters to `key` or `keys`.")
        if isinstance(keys, str):
            keys = [keys]
            values = [values]
        else:
            if len(keys) != len(values):
                raise ValueError("Length of `keys` does not equal to that of `values`.")
        # Check input
        for _key in keys:
            _folders = _key.split('.')
            for _folder in _folders:
                if len(_folder) < 1:
                    raise ValueError(f"Invalid key {_key}.")
        # Update local config
        for _key in keys:
            if _key in self._localConfig:
                # Update the local config if exists
                self._localConfig[keys] = values
        # Then update the remote config
        return self._addConfigToServer(keys, values)

    def getConfig(self, keys: Union[str, List[str]] = None, timestamp: int = None, ignoreCache: bool = False,
                  hierachy: bool = False) -> Union[Dict[str, Any], Any]:
        """
        Generate the pulse parameters for the experiment.

        :param keys: obtain a batch of configs
        :param timestamp: indicates the time stamp of the config, when it is None, ConfigServer will return the latest
            config.
        :param ignoreCache: ignore the local cache
        :param hierachy: the config will returned by hierachy dictionary if True.
        :return: the value of the config
        """
        # Check input parameters
        returnDict = {}
        readFromServerKeyList = []
        if keys is None or len(keys) < 1:
            pass
        else:
            # Initialize the dictionary
            readFromServerKeyList = []
            if isinstance(keys, str):
                keys = [keys]
            # Read from local config
            for key in keys:
                if key in self._localConfig.keys():
                    returnDict[key] = self._localConfig[key]
                else:
                    readFromServerKeyList.append(key)

        # Read from remote server
        fromServer = self._readFromServer(readFromServerKeyList, timestamp, ignoreCache)
        for keyFromServer in fromServer.keys():
            if hierachy:
                keyDict = keyFromServer.split('.')
                depth = len(keyDict)
                subDict = returnDict
                for _keyIdx, _key in enumerate(keyDict):
                    if _keyIdx == depth - 1:
                        subDict[_key] = fromServer[keyFromServer]
                    else:
                        if _key not in subDict.keys():
                            subDict[_key] = {}
                        subDict = subDict[_key]
            else:
                returnDict[keyFromServer] = fromServer[keyFromServer]
        if keys is not None and len(keys) == 1:
            return returnDict[keys[0]]
        else:
            return returnDict

    def labSpaceList(self, timeRange: List[int] = None) -> List[Any]:
        """
        Obtain the full labSpace list on server.

        :param timeRange: the time range list, the format is `[start_timestamp, end_timestamp]`.
        :return: the full list of the labSpaces
        """
        if timeRange is None:
            timeRange = [int(time.time()) - 3600 * 24 * 365, int(time.time())]
        return self._obtainLabSpaceList([int(timeRange[0]), int(timeRange[1])])

    def createLabSpace(self, labSpaceID: str, templateLabSpaceID: str = None, description: str = None) -> Dict:
        """
        Create a new labSpace to server.

        :param labSpaceID: the labSpace ID of the new labSpace
        :param templateLabSpaceID: the new labSpace will copy from an existing labSpace if this parameter is given
        :param description: the description of the labSpace
        :return: information about the new labSpace
        """
        return self._createLabSpaceToServer(labSpaceID, templateLabSpaceID, description)

    @property
    def qubits(self) -> Dict[int, str]:
        """ Get the qubits dictionary """
        return copy.deepcopy(self._QubitMapId2Label)

    @property
    def qubitIndexes(self) -> Dict[str, int]:
        """ Get the qubit-to-index dictionary """
        return copy.deepcopy(self._QubitMapLabel2Id)

    @property
    def channels(self) -> Dict[str, Any]:
        """ Get the qubit-to-index dictionary """
        return copy.deepcopy(self._channels)

    def _setChannels(self, channels: Dict[str, Any]):
        """
        Set the channel settings.
        """
        raise NotImplementedError("Abstract method `_setChannels()` is not implemented!")

    def _setQubits(self, qubits: List[str]):
        """
        Set the qubit names.
        """
        raise NotImplementedError("Abstract method `_setQubits()` is not implemented!")

    def getQubitLabel(self, qubitId: Union[List[int], int]):
        """
        Get the qubit label(s) according to the qubit index(es).
        """
        if isinstance(qubitId, int):
            for _label in self._QubitMapLabel2Id.keys():
                if self._QubitMapLabel2Id[_label] == qubitId:
                    return _label
            raise Error.ArgumentError(f"Qubit is not found: {qubitId}.")
        else:
            qubitLabelList = []
            for _label in self._QubitMapLabel2Id.keys():
                if self._QubitMapLabel2Id[_label] in qubitId:
                    qubitLabelList.append(_label)
            return qubitLabelList

    def getQubitIndex(self, qubitLabel: Union[List[str], str]):
        """
        Get the qubit name(s) according to the qubit index(es).
        """
        if isinstance(qubitLabel, str):
            for _label in self._QubitMapLabel2Id.keys():
                if _label == qubitLabel:
                    return self._QubitMapLabel2Id[_label]
            raise Error.ArgumentError(f"Qubit is not found: {qubitLabel}.")
        else:
            qubitLabelList = []
            for _label in self._QubitMapLabel2Id.keys():
                if _label in qubitLabel:
                    qubitLabelList.append(self._QubitMapLabel2Id[_label])
            return qubitLabelList
