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
Runner interface for superconducting system experiments
"""

from typing import Dict, Any, Set, List

from Quanlse.QWaveform import QResult, QJob
from Quanlse.Utils.ControlGate import ControlGate


class Runner(object):
    """
    The interface for quantum devices (local/remote simulator or physical devices). This class provides an interface
    to communicate with the devices. Users can overwrite this class to implement specific functions.
    """

    def __init__(self):
        """ Initialize the DataInterface """
        pass

    def run(self, job: QJob, measure: Set[int], ctrlGates: List[ControlGate] = None,
            conf: Dict[Any, Any] = None) -> QResult:
        """
        Generate the pulse parameters for the experiment.

        :param job: the QJob instance which contains the pulse definition
        :param ctrlGates: the list of control gates
        :param measure: the qubit index to perform measurement
        :param conf: the user-defined options/configurations.
        :return: the value of the config
        """
        raise NotImplementedError("Abstract method `run()` is not implemented!")
