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
Quantum executor interface definition
"""
import json
from typing import Dict, Optional

from Quanlse.Define import sdkVersion
from Quanlse import Algorithm, HardwareImplementation, TStepRange
from Quanlse.QPlatform.Utilities import numpyMatrixToDictMatrix, dictMatrixToNumpyMatrix


class QPulseResult:
    """
    The result of quantum pulse.
    """

    sdkVersion = sdkVersion
    """
    SDK Version from Define.sdkVersion
    """

    scheduler = None  # type: Optional[Dict]
    """
    scheduler
    """

    benchmark = None  # type: Optional[Dict]
    """
    benchmark
    """

    startTimeUtc = ''
    """
    start utc time
    """

    endTimeUtc = ''
    """
    end utc time
    """

    code = 0
    output = ''

    def __init__(self):
        self.sdkVersion = sdkVersion
        self.scheduler = None  # type: Optional[Dict]
        self.benchmark = None  # type: Optional[Dict]
        self.startTimeUtc = ''
        self.endTimeUtc = ''

    def fromJson(self, text: str):
        """
        fromJson
        """
        data = json.loads(text)
        if 'sdkVersion' in data:
            self.sdkVersion = data['sdkVersion']
        if 'scheduler' in data:
            self.scheduler = data['scheduler']
        if 'benchmark' in data:
            data['benchmark']['unitary'] = dictMatrixToNumpyMatrix(data['benchmark']['unitary'], complex)
            self.benchmark = data['benchmark']
        if 'startTimeUtc' in data:
            self.startTimeUtc = data['startTimeUtc']
        if 'endTimeUtc' in data:
            self.endTimeUtc = data['endTimeUtc']

    def toJson(self):
        """
        toJson
        """
        self.benchmark['unitary'] = numpyMatrixToDictMatrix(self.benchmark['unitary'])
        return json.dumps({
            'sdkVersion': self.sdkVersion,
            'scheduler': self.scheduler,
            'benchmark': self.benchmark,
            'startTimeUtc': self.startTimeUtc,
            'endTimeUtc': self.endTimeUtc,
        })


class QPulseImplement:
    """
    Implement params for quantum pulse.
    """

    program = None  # type: Dict
    """
    Json format of the circuit for pulse
    """

    algorithm = None  # type: Algorithm
    """
    Algorithm
    """

    hardwareImplementation = None  # type: HardwareImplementation
    """
    HardwareImplementation
    """

    tStep = TStepRange[0]  # type: float
    """
    TStep
    """

    def commit(self):
        """
        Commit task
        """
        pass
