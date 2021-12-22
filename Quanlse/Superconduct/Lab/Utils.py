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
Utils for superconducting system experiments
"""

import json
import copy
import pickle
import base64
from typing import List, Dict, Union

import numpy

from Quanlse.QPlatform import Error


class Scan(object):
    """
    Define the scanning range for superconducting experiment.
    """
    def __init__(self):
        """ Initialize """
        self._availableMode = ["linear", "seq"]
        self._mode = "linear"  # type: str
        self.start = 0.  # type: float
        self.stop = 0.  # type: float
        self.steps = 0  # type: int
        self.seq = []  # type: List[float]

    @property
    def mode(self) -> str:
        """ Scan mode """
        return self._mode

    @mode.setter
    def mode(self, modeStr: str):
        """ Scan mode """
        if modeStr in self._availableMode:
            self._mode = modeStr
        else:
            raise Error.ArgumentError(f"Scan mode {modeStr} is not available!")

    def getList(self):
        """
        Return the scanning list
        """
        if self._mode == 'linear':
            return numpy.linspace(self.start, self.stop, self.steps)

    def dump2Dict(self) -> Dict:
        """
        Return the dictionary which characterizes the current instance.
        """
        jobDict = {"mode": self._mode, "start": self.start, "stop": self.stop, "steps": self.steps, "seq": self.seq}
        return jobDict

    def dump2Json(self):
        """
        Return the JSON string which characterizes the current instance.
        """
        return json.dumps(self.dump2Dict())

    @staticmethod
    def parseJson(jsonObj: Union[str, Dict]) -> 'Scan':
        """
        Create object from a JSON encoded string or dictionary.

        :param jsonObj: a JSON encoded string or dictionary
        :return: a QJob object
        """
        if isinstance(jsonObj, str):
            jsonDict = json.loads(jsonObj)
        else:
            jsonDict = copy.deepcopy(jsonObj)
        # Create scan
        scanObj = Scan()
        scanObj.mode = jsonDict["mode"]
        scanObj.start = jsonDict["start"]
        scanObj.stop = jsonDict["stop"]
        scanObj.steps = jsonDict["steps"]
        scanObj.seq = jsonDict["seq"]
        return scanObj

    def dump(self) -> str:
        """
        Return a base64 encoded string.

        :return: encoded string
        """
        obj = copy.deepcopy(self)
        # Serialization
        byteStr = pickle.dumps(obj)
        base64str = base64.b64encode(byteStr)
        del obj
        return base64str.decode()

    @staticmethod
    def load(base64Str: str) -> 'Scan':
        """
        Create object from base64 encoded string.

        :param base64Str: a base64 encoded string
        :return: ExperimentScan object
        """
        byteStr = base64.b64decode(base64Str.encode())
        obj = pickle.loads(byteStr)  # type: Scan

        return obj


def scanLinear(start: float, stop: float, steps: int) -> Scan:
    """
    Define scanning range for experiment,
    """
    scan = Scan()
    scan.mode = "linear"
    scan.start = start
    scan.stop = stop
    scan.steps = steps
    return scan


def scanSeq(seq: List[float]):
    """
    Define scanning range for experiment,
    """
    scan = Scan()
    scan.mode = "seq"
    scan.seq = seq
    return scan
