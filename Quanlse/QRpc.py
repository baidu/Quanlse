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
Pulse RPC
"""
import os
from pprint import pprint
from enum import Enum
import json

from Quanlse.Define import sdkVersion, taskSourceQuanlse, circuitPackageFile, outputPath
from Quanlse.Define.Settings import outputInfo
from Quanlse.QPlatform import Error

from Quanlse.QTask import (
    invokeBackend,
    _retryWhileNetworkError,
    _uploadCircuit,
    _waitTask,
)

@_retryWhileNetworkError
def _createTask(token, circuitId, optimizer, backendParam=None, modules=[], debug=False, taskType="quanlse_optimizer"):
    """
    Create a task from the code
    """

    task = {
        "token": token,
        "circuitId": circuitId,
        "taskType": taskType,
        "sdkVersion": sdkVersion,
        "source": taskSourceQuanlse,
        "optimizer": optimizer,
        "modules": modules
    }

    if debug:
        task['debug'] = debug

    if backendParam:
        paramList = []
        for param in backendParam:
            if isinstance(param, Enum):
                paramList.append(param.value)
            else:
                paramList.append(param)
        task['backendParam'] = paramList

    ret = invokeBackend(
        "task/createTask",
        task
    )

    return ret['taskId']

def rpcCall(optimizer, args, kwargs, debug=False):
    """
    invoke remote optimizer
    """
    circuitId = None
    taskId = None

    program = {
        "optimizer": optimizer,
        "args": args,
        "kwargs": kwargs,
    }

    programBuf = json.dumps(program)
    with open(circuitPackageFile, 'wb') as file:
        file.write(programBuf.encode("utf-8"))

    token, circuitId = _uploadCircuit(circuitPackageFile)
    taskType = "quanlse_simulator" if optimizer == "runHamiltonian" else "quanlse_optimizer"
    taskId = _createTask(token, circuitId, optimizer, debug, taskType=taskType)

    if outputInfo:
        print(f"Circuit uploaded successfully, circuitId => {circuitId} taskId => {taskId}")

    taskResult = _waitTask(token, taskId, downloadResult=True)

    if type(taskResult) == str:
        raise Error.RuntimeError(taskResult)

    errorMsg = taskResult.get("outer_error")
    if errorMsg:
        raise Error.RuntimeError(errorMsg)

    else:
        localFile = os.path.join(outputPath, f'remote.{taskId}.origin.json')
        with open(localFile, "rb") as fObj:
            origin = json.loads(fObj.read())
            # return createFromJson(origin["qam"]), origin["infidelity"]
            return origin
