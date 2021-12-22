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
Agent for superconducting simulator
"""

from typing import Dict

from Quanlse.Superconduct.Simulator.PulseModel import PulseModel
from Quanlse.Superconduct.Simulator.Utils import SimulatorLabSpace, SimulatorRunner, SimulatorGenerator
from Quanlse.Superconduct.Lab.Experiment import Experiment
from Quanlse.Scheduler import Scheduler
from Quanlse.QPlatform import Error


class SimulatorAgent(object):
    """
    The agent for simulator.
    """
    def __init__(self, model: PulseModel, labSpace: SimulatorLabSpace, runner: SimulatorRunner,
                 generator: SimulatorGenerator):
        """ Initialization """
        # Basic instances
        self.model = model  # type: PulseModel
        self.labSpace = labSpace  # type: SimulatorLabSpace
        self.runner = runner  # type: SimulatorRunner
        self.generator = generator  # type: SimulatorGenerator

    def __call__(self, *args, **kwargs):
        """
        Return the qubit indexes of labels.
        """
        if isinstance(args, str):
            return self.labSpace.getQubitIndex(args)
        else:
            if len(args) > 1:
                return self.labSpace.getQubitIndex(list(args))
            else:
                return self.labSpace.getQubitIndex(args[0])

    def q(self, *args) -> int:
        """
        Return the sub system index in QJob instance.
        """
        if len(args) == 1:
            return self.labSpace.getQubitIndex(args[0])
        else:
            return self.labSpace.getQubitIndex(list(args))

    def qLabel(self, qubitIndex: int) -> int:
        """
        Return the sub system label in by QJob instance.
        """
        return self.labSpace.getQubitLabel(qubitIndex)

    def createExperiment(self) -> Experiment:
        """
        Create an Experiment instance
        """
        if self.labSpace is None:
            raise Error.RuntimeError("LabSpace is not set.")
        if self.runner is None:
            raise Error.RuntimeError("Runner is not set.")

        expObj = Experiment(self.labSpace, self.runner, self.generator)
        return expObj

    def createScheduler(self) -> Scheduler:
        """
        Create a Scheduler instance
        """
