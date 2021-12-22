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
ODE Solver
"""

import time
import copy
from numpy import ndarray, shape, identity, kron, dot, square, round, reshape, array
from typing import Dict, Any

from Quanlse.Utils.Functions import dagger
from Quanlse.Utils.Infidelity import isRho
from Quanlse.QPlatform import Error


def _cacheDictToList(ctrlCache: Dict[str, Any], waveCache: Dict[str, Any]):
    """
    Convert the dict type cache to the list type.
    """
    # Initialize ctrlList and waveList
    _ctrlList, _waveList = [], []
    for ctrlKey in ctrlCache:
        _ctrlList.append(ctrlCache[ctrlKey])
        _waveList.append(array(waveCache[ctrlKey]))
    return _ctrlList, _waveList


def solverNormal(ham: 'QHamiltonian', state0=None, shot=None, recordEvolution=False, accelerate=None):
    """
    Calculate the unitary evolution operator with a given Hamiltonian. This function supports
    both single-job and batch-job processing.

    :param ham: QHamiltonian object
    :param state0: the initial state vector. If None is given, this function will return the time-ordered
                   evolution operator, otherwise returns the final state vector.
    :param recordEvolution: the detailed procedure will be recorded if True
    :param accelerate: indicates the accelerator
    :param shot: return the population of the eigenstates when ``shot`` is provided
    :return: result dictionary (or a list of result dictionaries when ``jobList`` is provided)
    """

    # Import expm
    if accelerate:
        try:
            if accelerate == 'numba':
                from Quanlse.Utils.NumbaSupport import expm
            elif accelerate == 'jax':
                from Quanlse.Utils.JaxSupport import expm
            else:
                raise Error.Error(f"Unsupported accelerator {accelerate}!")
        except ImportError:
            raise Error.Error("You should install Numba (or Jax) to activate the acceleration.")
    else:
        from scipy.linalg import expm

    # Create cache
    if ham.driftCache is None:
        ham.buildCache()

    sysLevelLen = 1 if isinstance(ham.sysLevel, int) else len(ham.sysLevel)
    sysLevel = [ham.sysLevel] if isinstance(ham.sysLevel, int) else ham.sysLevel
    subSysNum = ham.subSysNum

    ctrlList, waveList = _cacheDictToList(ham.ctrlCache, ham.waveCache)

    if state0 is not None:
        # Initialize stateOrUnitary
        stateOrUnitary = copy.deepcopy(state0)
    else:
        # If the sysLevel is an integer to indicate the same energy level for each subsystem
        if sysLevelLen == 1:
            stateOrUnitary = identity(sysLevel[0] ** subSysNum, dtype=complex)
        else:
            # If the sysLevel is a list of energy levels for different subsystem
            dim = 1
            for subSysLvl in sysLevel:
                dim = dim * subSysLvl
            stateOrUnitary = identity(dim, dtype=complex)

    # Start time
    startTime = time.time()
    # Create evolution history list
    history = []

    for nowDt in range(0, ham.job.endTimeDt):
        totalHam = ham.driftCache.copy()
        # Traverse all the control Hamiltonian
        for key in range(len(ctrlList)):
            totalHam += ctrlList[key] * waveList[key][nowDt]
        tUnitary = expm(- 1j * totalHam * ham.dt)
        stateOrUnitary = dot(tUnitary, stateOrUnitary)
        if recordEvolution:
            history.append(stateOrUnitary)

    if state0 is not None:
        _state = stateOrUnitary
        _unitary = None
        if shot is not None:
            _population = list(map(int, round(square(abs(_state)) * shot)))
        else:
            _population = None
    else:
        _unitary = stateOrUnitary
        _state = None
        _population = None

    ret = {
        "unitary": _unitary,
        "state": _state,
        "population": _population,
        "dt": ham.dt,
        "dts": ham.job.endTimeDt,
        "ns": ham.job.endTimeDt * ham.dt,
        "time_consuming": time.time() - startTime,
        "evolution_history": history,
        "numba-simulator": "local-built-in-normal"
    }
    return ret


def solverAdaptive(ham: 'QHamiltonian', state0: ndarray = None, shot=None, tolerance: float = 0.01, accelerate=False):
    """
    Run the program calculating the unitary evolution operator for Hamiltonian.
    This is the adaptive algorithm for piecewise-constant, using the pulse sequences given in `hamiltonian`.
    In this algorithm, it applies the strategy of adaptive-step to accelerate the calculation.

    :param ham: QHamiltonian object
    :param state0: the initial state vector. If None is given, this function will return the time-ordered
                   evolution operator, otherwise returns the final state vector.
    :param shot: return the population of the eigenstates when ``shot`` is provided
    :param tolerance: the greatest error for approximation
    :param accelerate: indicates the accelerator
    :return: Return a dictionary containing the result.
    """

    # Import expm
    if accelerate:
        try:
            if accelerate == 'numba':
                from Quanlse.Utils.NumbaSupport import expm
            elif accelerate == 'jax':
                from Quanlse.Utils.JaxSupport import expm
            else:
                raise Error.Error(f"Unsupported accelerator {accelerate}!")
        except ImportError:
            raise Error.Error("You should install Numba (or Jax) to activate the acceleration.")
    else:
        from scipy.linalg import expm

    # Create cache
    if ham.driftCache is None:
        ham.buildCache()

    sysLevelLen = 1 if isinstance(ham.sysLevel, int) else len(ham.sysLevel)
    sysLevel = [ham.sysLevel] if isinstance(ham.sysLevel, int) else ham.sysLevel
    subSysNum = ham.subSysNum
    maxDt = ham.job.endTimeDt

    if state0 is not None:
        # Initialize stateOrUnitary
        stateOrUnitary = copy.deepcopy(state0)
    else:
        # If the sysLevel is an integer to indicate the same energy level for each subsystem
        if sysLevelLen == 1:
            stateOrUnitary = identity(sysLevel[0] ** subSysNum, dtype=complex)
        else:
            # If the sysLevel is a list of energy levels for different subsystem
            dim = 1
            for subSysLvl in sysLevel:
                dim = dim * subSysLvl
            stateOrUnitary = identity(dim, dtype=complex)

    # Start time
    startTime = time.time()

    # Start calculation
    calcCount = 0
    nowDt = 0
    while nowDt < maxDt:
        totalHam = ham.driftCache.copy()
        # Decide the step of adaptive algorithm
        maxChange = 0
        forwardStep = 1
        while nowDt < maxDt - forwardStep and maxChange < tolerance:
            for key in ham.ctrlCache:
                currentSeq = ham.waveCache[key][nowDt]
                nextSeq = ham.waveCache[key][nowDt + forwardStep]
                change = abs(nextSeq - currentSeq)
                if change > maxChange:
                    maxChange = change
            forwardStep += 1
        forwardStep -= 1
        forwardStep = max(1, forwardStep)
        # Calculate the total Hamiltonian
        for key in ham.ctrlCache:
            averageSequence = 0
            for innerDt in range(nowDt, nowDt + forwardStep):
                # Obtain the sequence
                currentSequence = ham.waveCache[key][innerDt]
                averageSequence += currentSequence
            totalHam += ham.ctrlCache[key] * averageSequence / forwardStep

        calcCount += 1
        tUnitary = expm(- 1j * totalHam * ham.dt * forwardStep)

        nowDt += forwardStep
        stateOrUnitary = dot(tUnitary, stateOrUnitary)

    if state0 is not None:
        _state = stateOrUnitary
        _unitary = None
        if shot is not None:
            _population = list(map(int, round(square(abs(_state)) * shot)))
        else:
            _population = None
    else:
        _unitary = stateOrUnitary
        _state = None
        _population = None

    ret = {
        "unitary": _unitary,
        "state": _state,
        "population": _population,
        "dt": ham.dt,
        "dts": ham.job.endTimeDt,
        "ns": ham.job.endTimeDt * ham.dt,
        "time_consuming": time.time() - startTime,
        "evolution_history": None,
        "numba-simulator": "local-built-in-adaptive"
    }
    return ret


def solverOpenSystem(ham: 'QHamiltonian', state0=None, recordEvolution=False, accelerate=False):
    """
    Calculate the unitary evolution operator with a given Hamiltonian. This function supports
    both single-job and batch-job processing.

    :param ham: QHamiltonian object
    :param state0: the initial state vector. If None is given, this function will return the time-ordered
                   evolution operator, otherwise returns the final state vector.
    :param recordEvolution: the detailed procedure will be recorded if True
    :param accelerate: indicates the accelerator
    :return: result dictionary (or a list of result dictionaries when ``jobList`` is provided)
    """
    if accelerate:
        try:
            if accelerate == 'numba':
                from Quanlse.Utils.NumbaSupport import expm
            elif accelerate == 'jax':
                from Quanlse.Utils.JaxSupport import expm
            else:
                raise Error.Error(f"Unsupported accelerator {accelerate}!")
        except ImportError:
            raise Error.Error("You should install Numba (or Jax) to activate the acceleration.")
    else:
        from scipy.linalg import expm

    if state0.shape[1] == 1:
        rho0 = state0 @ dagger(state0)
    else:
        if isRho(state0):
            rho0 = state0
        else:
            raise Error.ArgumentError('The input state is neither a density matrix nor a state vector')

    ctrlList, waveList = _cacheDictToList(ham.ctrlCache, ham.waveCache)

    startTime = time.time()
    dt = ham.dt
    qubitsNum = ham.subSysNum

    # Cache information
    ld = ham.dissipationSuperCache
    maxDt = ham.job.endTimeDt
    maxNs = ham.job.endTime
    sysLevel = ham.sysLevel

    # Initialization
    evolutionHistory = []
    propHistotry = []
    dim = None

    # Calculate the dimension
    if isinstance(sysLevel, int):
        dim = sysLevel ** qubitsNum
    elif isinstance(sysLevel, list):
        dim = 1
        for i in sysLevel:
            dim = dim * i

    idn = identity(dim, dtype=complex)
    rhoDim = shape(rho0)[0]

    if rhoDim != dim:
        raise Error.ArgumentError(
            f'The dimension of the input density matrix {rhoDim} is not matched with system level {dim}.')

    _rho = copy.deepcopy(rho0)
    evolutionHistory.append(_rho)
    propt = identity(dim * dim, dtype=complex)

    for nowDt in range(0, maxDt):
        # Initialization
        _ham = ham.driftCache.copy()
        for key in range(len(ctrlList)):
            _ham += ctrlList[key] * waveList[key][nowDt]
        lh = kron(_ham, idn) - kron(idn, _ham.T)
        lind = -1j * lh + ld
        # calculate the propagator
        prop = expm(lind * dt)
        # convert the density matrix to a state vector
        _rhoVec = reshape(_rho, (-1, 1))
        # update the state vector using propagator
        _rhoVec = prop @ _rhoVec
        # convert the state vector back to the density matrix
        _rho = reshape(_rhoVec.T, (dim, dim))
        propt = prop @ propt
        if recordEvolution:
            evolutionHistory.append(_rho)
            propHistotry.append(prop)

    ret = {
        "propagator": propt,
        "state": _rho,
        "dt": ham.dt,
        "dts": maxDt,
        "ns": maxNs,
        "time_consuming": time.time() - startTime,
        "propagator_history": propHistotry,
        "evolution_history": evolutionHistory,
        "numba-simulator": "local-built-in-open-sys"
    }
    return ret
