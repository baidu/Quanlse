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
Pipeline class for Quanlse Scheduler.
"""

import copy
from typing import List, Union, Dict, Any, Callable

from Quanlse.QWaveform import QJob
from Quanlse.QPlatform.Error import ArgumentError


class SchedulerProcess:
    """
    Scheduler process for SchedulerPipeline instances to perform scheduling strategy.

    :param title: a user-given title to the pi object
    :param processFunc: Callable function to perform scheduling strategies
    """

    def __init__(self, title: str, processFunc: Callable):
        """
        Constructor for the SchedulerProcess class.
        """
        self.title = title
        self._processFunc = processFunc
        self._conf = {}  # type: Dict[str, Any]

    def __call__(self, job: QJob, scheduler: 'Scheduler') -> QJob:
        """
        Call the scheduling function.
        """
        if self._processFunc is None:
            raise ArgumentError("Scheduling function is not set.")
        return self._processFunc(job=job, scheduler=scheduler)


class SchedulerPipeline:
    """
    SchedulerPipeline instance is a sequence of SchedulerProcess to perform scheduling strategies.

    :param title: a user-given title to the pi object
    """

    def __init__(self, title: str = ""):
        """
        Initializing
        """
        self._title = title
        self._pipeline = []  # type: List[SchedulerProcess]

    def __call__(self, job: QJob, scheduler: 'Scheduler') -> QJob:
        """
        Call the instance and run the pipeline in order,
        and output a QJob instance.
        """
        for idx, process in enumerate(self._pipeline):
            if isinstance(process, Callable):
                job = process(job, scheduler)
            else:
                raise ArgumentError("Scheduler process is not callable.")
        return job

    @property
    def pipeline(self) -> List[SchedulerProcess]:
        """
        Return jobs.
        """
        return self._pipeline

    def addPipelineJob(self, process: Union[SchedulerProcess, List[SchedulerProcess]]) -> None:
        """
        Add pipeline job in to the job list
        """
        if isinstance(process, List):
            for pl in process:
                self._pipeline.append(pl)
        elif isinstance(process, SchedulerProcess):
            self._pipeline.append(process)

    def __add__(self, other: Union[SchedulerProcess, List[SchedulerProcess], 'SchedulerPipeline']) \
            -> 'SchedulerPipeline':
        """ Add  """
        if isinstance(other, SchedulerProcess):
            # Add a Pipeline job in to the list
            self.addPipelineJob(other)
            return self
        elif isinstance(other, SchedulerPipeline):
            for job in other.pipeline:
                self.addPipelineJob(copy.deepcopy(job))
            return self
