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
Default SchedulerPipeline for QuanlseSchedulerSuperconduct.
"""

from numpy import array

from Quanlse.Scheduler.SchedulerPipeline import SchedulerProcess, toLayer, fillUp, findMaxT, addToJob
from Quanlse.Scheduler import Scheduler


def _leftAlignedCore(scheduler: Scheduler) -> None:
    """
    The left alignment strategy for scheduling.

    :param scheduler: Scheduler object containing the circuit information
    """

    # Initialize layers
    layer = []
    for _ in range(scheduler.subSysNum):
        layer.append([])

    # First convert gates in Scheduler to layers
    toLayer(layer, scheduler)
    fillUp(layer)

    # Transpose layers
    layer = array(layer).T.tolist()

    # Find max time for each layer
    maxT = findMaxT(layer)

    # add waves to job
    addToJob(layer=layer, scheduler=scheduler, maxT=maxT)


leftAligned = SchedulerProcess("LeftAligned", _leftAlignedCore)
"""
A SchedulerProcess instance containing the left-aligned scheduling strategy.
"""
