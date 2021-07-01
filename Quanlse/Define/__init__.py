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
Global Definitions
"""

import os
import sys

import Quanlse
from Quanlse.QPlatform import Error

env = 'prod'

"""
Environment

Do not modify.

Values: 'prod', 'test'

Used to create or test environment.
"""
if env == "test":
    raise Error.RuntimeError('Not implemented')
else:
    # service address for production
    quantumHubAddr = 'https://quantum-hub.baidu.com/api'
    quantumBucket = 'quantum-task'

sdkVersion = 'Quanlse 2.0.0'
"""
SDK Version

Do not modify.

Used for task submission.
"""

hubToken = os.environ.get('HUBTOKEN', '')
"""
Hub Token

Do not modify.

Used for Quanlse cloud task.

Users can acquire tokens from http://quantum-hub.baidu.com .

Token Management -> Create/View Token
"""

taskSource = os.environ.get('SOURCE', 'PySDK')
taskSourceQuanlse = os.environ.get('SOURCE', 'QuanlseSDK')
"""
Task Source

Do not modify.

Values: 'PySDK', 'PyOnline'

PySDK or PyOnline.
"""

noLocalTask = os.environ.get('NOLOCALTASK', None)
"""
No Local Task

Do not modify.

Values: None or Other

Used for PyOnline.
"""

pollInterval = 5
"""
Poll interval in seconds

Do not modify.

Used for task check.
"""

waitTaskRetrys = 10
"""
Wait task retrys

Do not modify.

Retry count for waiting task in case network failed.
"""

outputPath = os.path.abspath(os.path.join(Quanlse.__file__, '../../Output'))
"""
Output Path

Do not modify by user.

Will be created, when not exist.
"""
if 'sphinx' in sys.modules:
    outputPath = ''
else:
    os.makedirs(outputPath, mode=0o744, exist_ok=True)

circuitPackageFile = os.path.join(outputPath, 'Package.pb')
"""
Circuit Package File

Do not modify.

Circuit hdf5 target file
"""
if 'sphinx' in sys.modules:
    circuitPackageFile = ''