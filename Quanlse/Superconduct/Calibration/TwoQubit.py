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
Two-qubit gates Calibration
"""

import numpy
from math import pi
from numpy import array
from typing import Callable, Any, List, Optional

from Quanlse.QOperation.RotationGate import RX
from Quanlse.QOperation.FixedGate import CZ, H
from Quanlse.QPlatform.Error import ArgumentError
from Quanlse.Scheduler import Scheduler
from Quanlse.Superconduct.Simulator import PulseModel
from Quanlse.Utils.Functions import getPopulationOnQubit, basis, project


def caliSingleQubitGatesJob(sche: Scheduler, q0: int, q1: int, q0Para: Optional[List[float]],
                            q1Para: Optional[List[float]], returnMatrix: bool = False):
    """
    Calibrate the dynamical phase by tuning the Z pulse.

    :param sche: the Scheduler instance
    :param q0: the index of the first qubit to be calibrated
    :param q1: the index of the second qubit to be calibrated
    :param q0Para: the pulse parameters for the first qubit (piAmp and DRAG coefficient)
    :param q1Para: the pulse parameters for the second qubit (piAmp and DRAG coefficient)
    :param returnMatrix: returns the ideal matrix for the circuit when True
    :return: the QJob instance
    """

    if q0Para is not None:
        sche.conf["caliDataXY"][q0]["piAmp"] = q0Para[0]
        sche.conf["caliDataXY"][q0]["dragCoef"] = q0Para[1]

    if q1Para is not None:
        sche.conf["caliDataXY"][q1]["piAmp"] = q1Para[0]
        sche.conf["caliDataXY"][q1]["dragCoef"] = q1Para[1]

    matrixList = []
    jobList = sche.ham.createJobList()

    # Job 1: two H gates run simultaneously
    sche.clearCircuit()
    H(sche.Q[q0])
    H(sche.Q[q1])
    jobList.addJob(sche.schedule())
    if returnMatrix:
        matrixList.append(sche.getMatrix())

    # Job 2: two X/2 gates run simultaneously
    sche.clearCircuit()
    RX(pi / 2)(sche.Q[q0])
    RX(pi / 2)(sche.Q[q1])
    jobList.addJob(sche.schedule())
    if returnMatrix:
        matrixList.append(sche.getMatrix())

    # Job 3: two X gates run simultaneously
    sche.clearCircuit()
    RX(pi)(sche.Q[q0])
    RX(pi)(sche.Q[q1])
    jobList.addJob(sche.schedule())
    if returnMatrix:
        matrixList.append(sche.getMatrix())

    # Job 4: one X gate runs on the first qubit
    sche.clearCircuit()
    RX(pi)(sche.Q[q0])
    jobList.addJob(sche.schedule())
    if returnMatrix:
        matrixList.append(sche.getMatrix())

    return jobList, matrixList


def caliSingleQubitGates(sche: Scheduler, q0: int, q1: int, runner: Callable = None, bounds: List = None,
                         q0ParaInit: Optional[List[float]] = None, q1ParaInit: List[float] = None, options: Any = None):
    """
    Find the best qubit frequency.

    :param sche: the Scheduler instance
    :param q0: the index of the first qubit to be calibrated
    :param q1: the index of the second qubit to be calibrated
    :param runner: a callable function which can pass the QJob/QJobList instances and obtain the result
    :param bounds: the optimization bounds for each parameter
    :param q0ParaInit: specify a fixed frequency shift for qubit 0
    :param q1ParaInit: the list of the qubit 1 frequency shift (based on the present qubit frequency)
    :param options: options pass to the runner function
    """

    _, idealMatrixList = caliSingleQubitGatesJob(sche, q0, q1, None, None, True)

    def _singleQubitCaliLoss(x):
        """
        Obtain the calibration pulse jobs and calculate the loss function.
        """

        # Generate the pulse jobs for calibration
        caliJobs, _ = caliSingleQubitGatesJob(sche, q0, q1, [x[0], x[1]], [x[2], x[3]], False)

        # Simulate the calibration pulse
        if isinstance(sche, PulseModel):
            results = sche.simulate(jobList=caliJobs, options=options)
        else:
            if runner is None or not callable(runner):
                raise ArgumentError("Please input a callable runner!")
            results = runner(jobList=caliJobs, options=options)

        lossSum = 0.
        for _jobIdx, _idealMat in enumerate(idealMatrixList):
            # Obtain the output population value
            pop = project(numpy.square(abs(results[_jobIdx]["state"])).T[0], sche.subSysNum, sche.sysLevel, 2)

            # Obtain the ideal population
            stateIdeal = _idealMat @ basis(_idealMat.shape[0], 0).T[0]
            popIdeal = numpy.square(abs(stateIdeal))

            # Calculate the loss function
            lossSum += numpy.sum(numpy.abs(pop - popIdeal)) / len(pop)

        aveLoss = lossSum / len(idealMatrixList)
        return aveLoss

    from scipy.optimize import minimize
    InitParas = [q0ParaInit[0], q0ParaInit[1], q1ParaInit[0], q1ParaInit[1]]
    opt = minimize(_singleQubitCaliLoss, x0=numpy.array(InitParas), bounds=bounds, method="l-bfgs-b")
    return opt["x"][0], opt["x"][1], opt["x"][2], opt["x"][3], opt["fun"]


def czCaliCondPhaseJob(sche: Scheduler, q0: int, q1: int, q0ZAmp: float, q1ZAmp: float, czLen: float):
    """
    Generate the calibration pulses (QJobList instances) for the conditional phase by tuning the Z pulse.

    :param sche: the Scheduler instance
    :param q0: the index of the first qubit to be calibrated
    :param q1: the index of the second qubit to be calibrated
    :param q0ZAmp: the amplitude of Z pulse on the first qubit
    :param q1ZAmp: the amplitude of Z pulse on the second qubit
    :param czLen: the duration of the Z pulse
    :return: the QJobList instance
    """

    qubits = (q0, q1)

    # Set the pulse parameter
    sche.conf["caliDataCZ"][qubits]["q0ZAmp"] = q0ZAmp
    sche.conf["caliDataCZ"][qubits]["q1ZAmp"] = q1ZAmp
    sche.conf["caliDataCZ"][qubits]["czLen"] = czLen
    sche.conf["caliDataCZ"][qubits]["q0VZPhase"] = 0.
    sche.conf["caliDataCZ"][qubits]["q1VZPhase"] = 0.

    # -------------------------------------
    # 1. When qubit B is initialized as |1>
    # -------------------------------------
    sche.clearCircuit()
    # Add the X pulse
    RX(pi / 2)(sche.Q[q0])
    RX(pi)(sche.Q[q1])
    # Add the Z pulse
    CZ(sche.Q[q0], sche.Q[q1])
    # Add the X/2 pulse
    RX(pi / 2)(sche.Q[q0])
    # Generate the pulse
    job1 = sche.schedule()

    # -------------------------------------
    # 2. When qubit B is initialized as |0>
    # -------------------------------------
    sche.clearCircuit()
    # Add the X pulse
    RX(pi / 2)(sche.Q[q0])
    # Add the Z pulse
    CZ(sche.Q[q0], sche.Q[q1])
    # Add the X/2 pulse
    RX(pi / 2)(sche.Q[q0])
    # Generate the pulse
    job2 = sche.schedule()

    # Insert into the jobList
    jobList = sche.ham.createJobList()
    jobList.addJob(job1)
    jobList.addJob(job2)

    return jobList


def czCaliCondPhase(sche: Scheduler, q0: int, q1: int, runner: Callable = None, method: str = 'dual_annealing',
                    maxIter: int = 50, q0ZAmpInit: float = None, q1ZAmpInit: float = None, czLenInit: float = None,
                    options: Any = None):
    """
    Run the conditional phase calibration procedure for the controlled-Z gates.

    :param sche: the Scheduler instance
    :param q0: the index of the first qubit to be calibrated
    :param q1: the index of the second qubit to be calibrated
    :param runner: a callable function which can pass the QJob/QJobList instances and obtain the result
    :param method: the optimization method, can be chose from 'dual_annealing' or 'Nelder-Mead'
    :param maxIter: the maximum iteration
    :param q0ZAmpInit: the initial value of the amplitude of Z pulse on the first qubit
    :param q1ZAmpInit: the initial value of the amplitude of Z pulse on the second qubit
    :param czLenInit: the initial value of the duration of Z pulse
    :param options: options pass to the runner function
    """

    def _condPhaseCaliLoss(x):
        """
        Obtain the calibration pulse jobs and calculate the loss function.
        """

        # Generate the pulse jobs for calibration
        caliJobs = czCaliCondPhaseJob(sche, q0, q1, x[0], x[1], x[2])

        # Simulate the calibration pulse
        if isinstance(sche, PulseModel):
            results = sche.simulate(jobList=caliJobs, options=options)
        else:
            if runner is None or not callable(runner):
                raise ArgumentError("Please input a callable runner!")
            results = runner(jobList=caliJobs, options=options)

        # Obtain the output population value
        popB1 = numpy.square(abs(results[0]["state"])).T[0]
        popB0 = numpy.square(abs(results[1]["state"])).T[0]

        # Population of the first qubit in basis |0> when the second qubit is |1>
        pop0B1 = getPopulationOnQubit(popB1, 1, sche.sysLevel)

        # Population of the first qubit in basis |1> when the second qubit is |0>
        pop1B0 = getPopulationOnQubit(popB0, 1, sche.sysLevel)

        # Calculate the loss function
        loss = pop0B1[0] + pop1B0[1]
        return loss

    if method == 'dual_annealing':
        from scipy.optimize import dual_annealing
        opt = dual_annealing(_condPhaseCaliLoss, [(-5, 5), (-5, 5), (30, 50)], maxiter=maxIter)
    elif method == 'Nelder-Mead':
        from scipy.optimize import minimize
        if czLenInit is None or q0ZAmpInit is None or q1ZAmpInit is None:
            raise ArgumentError(f"You must specify initial values when using '{method}'!")
        opt = minimize(_condPhaseCaliLoss, array([q0ZAmpInit, q1ZAmpInit, czLenInit]), method=method)
    else:
        raise ArgumentError(f"Unsupported optimization method '{method}'!")

    return opt["x"][0], opt["x"][1], opt["x"][2], opt["fun"]


def czCaliDynamicalPhaseJob(sche: Scheduler, q0: int, q1: int, q0VZPhase: float, q1VZPhase: float):
    """
    Calibrate the dynamical phase by tuning the Z pulse.

    :param sche: the Scheduler instance
    :param q0: the index of the first qubit to be calibrated
    :param q1: the index of the second qubit to be calibrated
    :param q0VZPhase: the amplitude of the calibration phase on the first qubit
    :param q1VZPhase: the amplitude of the calibration phase on the second qubit
    :return: the QJob instance
    """

    qubits = (q0, q1)

    # Set the pulse parameter
    sche.conf["caliDataCZ"][qubits]["q0VZPhase"] = q0VZPhase
    sche.conf["caliDataCZ"][qubits]["q1VZPhase"] = q1VZPhase

    # -------------------------------------
    # 1. When qubit B is initialized as |1>
    # -------------------------------------
    sche.clearCircuit()
    # Add the H pulse
    H(sche.Q[q0])
    H(sche.Q[q1])
    # Add the Z pulse
    CZ(sche.Q[q0], sche.Q[q1])
    # Add the X/2 pulse
    H(sche.Q[q1])
    # Generate the pulse
    job = sche.schedule()

    return job


def czCaliDynamicalPhase(sche: Scheduler, q0: int, q1: int, runner: Callable = None, method: str = 'Nelder-Mead',
                         maxIter: int = 50, q0VZPhaseInit: float = None, q1VZPhaseInit: float = None,
                         options: Any = None):
    """
    Run the dynamical phase calibration procedure for the controlled-Z gates.

    :param sche: the Scheduler instance
    :param q0: the index of the first qubit to be calibrated
    :param q1: the index of the second qubit to be calibrated
    :param runner: a callable function which can pass the QJob/QJobList instances and obtain the result
    :param method: the optimization method, can be chose from 'dual_annealing' or 'Nelder-Mead'
    :param maxIter: the maximum iteration
    :param q0VZPhaseInit: the initial value of the amplitude of Z pulse on the first qubit
    :param q1VZPhaseInit: the initial value of the amplitude of Z pulse on the second qubit
    :param options: options pass to the runner function
    """

    def _dynamicalPhaseCaliLoss(x):
        """
        Obtain the calibration pulse jobs and calculate the loss function.
        """

        # Generate the pulse jobs for calibration
        caliJob = czCaliDynamicalPhaseJob(sche, q0, q1, x[0], x[1])

        # Simulate the calibration pulse
        if isinstance(sche, PulseModel):
            results = sche.simulate(job=caliJob, options=options)
        else:
            if runner is None or not callable(runner):
                raise ArgumentError("Please input a callable runner!")
            results = runner(job=caliJob, options=options)

        # Obtain the output population value
        pop = project(numpy.square(abs(results[0]["state"])).T[0], sche.subSysNum, sche.sysLevel, 2)

        # Obtain the ideal population
        uIdeal = sche.getMatrix()
        stateIdeal = uIdeal @ basis(uIdeal.shape[0], 0).T[0]
        popIdeal = numpy.square(abs(stateIdeal))

        # Calculate the loss function
        loss = numpy.sum(numpy.abs(pop - popIdeal)) / len(pop)
        return loss

    if method == 'dual_annealing':
        from scipy.optimize import dual_annealing
        opt = dual_annealing(_dynamicalPhaseCaliLoss, [(-2 * pi, 2 * pi), (-2 * pi, 2 * pi)], maxiter=maxIter)
    elif method == 'Nelder-Mead':
        from scipy.optimize import minimize
        if q0VZPhaseInit is None or q1VZPhaseInit is None:
            raise ArgumentError(f"You must specify 'q0VZPhaseInit' and 'q1VZPhaseInit' when using '{method}'!")
        opt = minimize(_dynamicalPhaseCaliLoss, array([q0VZPhaseInit, q1VZPhaseInit]), method=method)
    else:
        raise ArgumentError(f"Unsupported optimization method '{method}'!")

    return opt["x"][0], opt["x"][1], opt["fun"]
