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
The `Lab` package is for rapidly designing superconducting experiments, including the scan or normal experiments;
it also provides the interface `Superconduct.Lab.LabSpace` to access the data service, `Superconduct.Lab.Runner`
to connect the experiment devices or simulator.
"""

from .LabSpace import LabSpace
from .Experiment import Experiment
from .ExperimentTask import ExperimentTask
from .Runner import Runner
from .Generator import Generator
from .Utils import Scan, scanLinear, scanSeq
