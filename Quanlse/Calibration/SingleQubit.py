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
Single-qubit Calibration
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from Quanlse.QOperator import driveX, driveZ
from Quanlse.QWaveform import gaussian, square
from Quanlse.Simulator import PulseModel
from Quanlse.Utils.Functions import basis, dagger, expect
from numpy import linspace, array
from math import pi
from typing import Union, List, Any, Optional
from Quanlse.Define import Error
from scipy import optimize


def ampRabi(pulseModel: PulseModel, pulseFreq: Union[int, float], ampRange: list, tg: int,
            sample: int = 100) -> [list, list]:
    """
    Perform a Rabi Oscillation by varying the pulse amplitudes. This function returns a list of amplitudes scanned
    and a list of populations.

    :param pulseModel: a pulseModel object
    :param pulseFreq: frequency of the pulse
    :param ampRange: a list of amplitude bounds
    :param tg: fixed pulse duration
    :param sample: sample size (default at 100)
    :return: a list of amplitudes scanned and a list of populations
    """
    # initialize population list and define amplitude list
    popList = []
    ampList = linspace(ampRange[0], ampRange[1], sample)

    # define job list
    qJobList = pulseModel.ham.createJobList()
    for amp in ampList:
        qJob = pulseModel.ham.createJob()
        qJob.setLO(driveX, 0, freq=pulseFreq)
        qJob.appendWave(driveX, 0, gaussian(tg, amp, tg / 2, tg / 8))
        qJobList.addJob(qJob)
    stateInit = basis(3, 0)

    # run simulation
    results = pulseModel.ham.simulate(jobList=qJobList, state0=stateInit)

    # obtain population from results
    for result in results:
        rho = result['state']
        prj = basis(3, 1) @ dagger(basis(3, 1))
        popList.append(expect(prj, rho))
    return list(ampList), popList


def tRabi(pulseModel: PulseModel, pulseFreq: Union[int, float], tRange: list, amp: float,
          sample: int = 100) -> [list, list]:
    """
    Perform a Rabi Oscillation by varying the pulse duration. This function returns a list of pulse duration scanned
    and a list of populations.

    :param pulseModel: a pulseModel object
    :param pulseFreq: frequency of the pulse
    :param amp: fixed pulse amplitude
    :param tRange: a list of duration bounds
    :param sample: sample size (default at 100)
    :return: a list of pulse duration scanned and a list of populations
    """

    # initialize population list and define time list
    popList = []
    tList = linspace(tRange[0], tRange[1], sample)

    # define job list
    qJobList = pulseModel.ham.createJobList()
    for t in tList:
        qJob = pulseModel.ham.createJob()
        qJob.setLO(driveX, 0, freq=pulseFreq)
        qJob.appendWave(driveX, 0, gaussian(t, amp, t / 2, t / 8))
        qJobList.addJob(qJob)
    stateInit = basis(3, 0)

    # run simulation
    results = pulseModel.ham.simulate(jobList=qJobList, state0=stateInit)

    # obtain population from results
    for result in results:
        rho = result['state']
        prj = basis(3, 1) @ dagger(basis(3, 1))
        popList.append(expect(prj, rho))
    return list(tList), popList


def fitRabi(popList: list, xList: list, guess: Optional[Union[List[Any], np.ndarray]] = None,
            sample: int = 5000, method: str = 'minimize') -> [float, float]:
    r"""
    Find pi-pulse duration/amplitude with Rabi results. This function takes a list of population, a list of whatever
    is scanned, sample size and a list of initial parameters for the fitting function. The fitting function takes
    form: :math:`y = a \cdot \cos(b \cdot x + c) + d`. The function returns the pi/2 and pi pulse amplitude(duration).

    :param popList: a list of population returned from the Rabi function.
    :param xList: a list of scanned variables(durations or amplitudes)
    :param guess: initial guess of the fitting function.
    :param sample: sample size
    :param method: method for optimization, 'minimize' or dual_annealing
    :return: the pi/2 and pi pulse amplitude(duration)
    """

    # Estimate the oscillation frequency from the given data

    def _estimateInit(_x, _y):
        yMax = max(_y)
        idx = find_peaks(_y, yMax / 2)[0][0]
        halfPeriod = _x[idx] - _x[0]
        period = 2 * halfPeriod
        freq = 1 / period
        aEst = -0.5
        bEst = 2 * pi * freq
        cEst = 0.
        dEst = 0.5
        estInit = np.array([aEst, bEst, cEst, dEst])
        return estInit

    # Define the function to be fitted
    def fit(x, a, b, c, d):
        return a * np.cos(b * x + c) + d

    # Define the loss function
    def loss(x):
        estList = np.array([fit(t, x[0], x[1], x[2], x[3]) for t in xList])
        _loss = np.sum(np.abs(estList - popList))
        return _loss

    # Fit the data
    if method == 'dual_annealing':
        opt = optimize.dual_annealing(loss, [(-100., 100.), (-100., 100.), (-100., 100.), (-100., 100.)], maxiter=4000)

    elif method == 'minimize':
        if guess is None:
            guess = _estimateInit(xList, popList)
            opt = optimize.minimize(loss, x0=guess)
        else:
            opt = optimize.minimize(loss, x0=guess)

    else:
        raise Error.ArgumentError('The method is \'dual_annealing\' or \'minimize\'.')

    def yFit(x):
        return fit(x, opt['x'][0], opt['x'][1], opt['x'][2], opt['x'][3])

    sampleList = np.linspace(xList[0], max(xList), sample)
    y = [yFit(x) for x in sampleList]
    peak = find_peaks(y)[0][0]
    x180 = sampleList[peak]
    x90 = x180 / 2

    return x90, x180


def longRelax(pulseModel: PulseModel, dt: float, pulseFreq: float, piAmp: float, piLen: float,
              maxIdle: float, initFit: list) -> [float, list, list, list]:
    r"""
    Find T1 using longitudinal relaxation, note that this function includes the fitting function.
    The fitting function takes form: :math:`y = \exp(-x / T_1 )`.

    :param pulseModel: a pulse model object
    :param dt: AWG sampling time
    :param pulseFreq: pulse frequency
    :param piAmp: pi-pulse amplitude
    :param piLen: pi-pulse length
    :param maxIdle: max idling time after the pi-pulse
    :param initFit: an initial guess of T1
    :return: predicted T1, a list of time, a list of population and a list of population on the fitted curve
    """

    # create job
    qJob = pulseModel.createQHamiltonian().createJob()
    qJob.setLO(driveX, 0, freq=pulseFreq)
    qJob.addWave(driveX, 0, gaussian(piLen, piAmp, piLen / 2, piLen / 8), t0=0.)
    qJob.addWave(driveX, 0, square(maxIdle, 0), t0=0.)
    stateInit = basis(3, 1)

    # run simulation
    results = pulseModel.ham.simulate(job=qJob, state0=stateInit, isOpen=True, recordEvolution=True)
    states = results[0]['evolution_history']

    # calculate populations from results
    popList = []
    for state in states:
        prj = basis(3, 1) @ dagger(basis(3, 1))
        popList.append(expect(prj, state))
    popList = popList[int(piLen / dt)::]
    tList = list(linspace(0, maxIdle, len(popList)))

    # fit curve
    T1 = fitT1(tList, popList, initFit)
    return T1[0], tList, popList, np.exp(-1. / T1[0] * np.array(tList))


def fitT1(tList: list, popList: list, init: list) -> list:
    r"""
    T1 fitting function, this function takes a list of times, a list of populations and an initial guess for T1.
    The fitting function takes form: :math:`y = \exp(-x / T_1 )`.

    :param tList: a list of times
    :param popList: a list of populations
    :param init: an initial guess
    :result: predicted T1
    """

    # Define fitting function
    def fit(x, t1):
        y = np.exp(-1. / t1 * np.array(x))
        return y

    # Fit curve
    paraFit, _ = curve_fit(fit, tList, popList, init)
    return list(paraFit)


def ramsey(pulseModel: PulseModel, pulseFreq: float, tg: float, x90: float, sample: int = 100,
           maxTime: int = 800, detuning: float = None) -> [list, list]:
    """
    Perform a Ramsey experiment. This function takes a PulseModel object, pulse frequency, pi/2 pulse length,
    pi/2 pulse amplitude, sample size, maximum idling time and detuning pulse amplitude. This function returns
    a list of idling time and a list of population.

    :param pulseModel: a PulseModel object
    :param pulseFreq: pulse frequency
    :param tg: pi/2 pulse length
    :param x90: pi/2 pulse amplitude
    :param sample: sample size
    :param maxTime: maximum idling time
    :param detuning: detuning amplitude
    """

    # get Hamiltonian from pulse model
    ham = pulseModel.createQHamiltonian(frameMode='lab')

    # Define the detuning
    if detuning is None:
        detuning = 2 * pi * 8. / maxTime

    # time list
    tList = np.linspace(0, maxTime, sample)

    # define jobList for Ramsey experiment
    ramseyJob = ham.createJobList()
    for t in tList:
        job = pulseModel.ham.createJob()
        job.setLO(driveX(3), 0, freq=pulseFreq)
        job.addWave(driveX(3), 0, gaussian(t=tg, a=x90 / 2, tau=tg / 2, sigma=tg / 8), t0=0.)  # pi/2 pulse
        job.addWave(driveZ(3), 0, waves=square(t=t, a=detuning), t0=tg)  # simulate the rotation due to the detuning
        job.addWave(driveX(3), 0, gaussian(t=tg, a=-x90 / 2, tau=tg / 2, sigma=tg / 8), t0=tg + t)  # pi/2 pulse
        ramseyJob.addJob(job)

    # run simulation
    stateInit = basis(3, 0)
    results = pulseModel.ham.simulate(state0=stateInit, jobList=ramseyJob, isOpen=True)

    # calculate populations from results
    popList = []
    for result in results:
        rho = result['state']
        prj = basis(3, 1) @ dagger(basis(3, 1))
        popList.append(expect(prj, rho))
    return list(tList), popList


def fitRamsey(t1: float, popList: list, tList: list, detuning: float) -> [float, list]:
    r"""
    Find T2 from Ramsey's result. This function takes a estimated T1 value, a list of population, a list
    of idling time and the amplitude of the detuning. The fitting function takes form:
    :math:`y = - 0.5 \cdot \cos(a \cdot x) \exp(-b \cdot x) + 0.5`

    :param t1: estimated T1.
    :param popList: a list of population from the Ramsey experiment
    :param tList: a list of idling time from the Ramsey experiment
    :param detuning: detuning amplitude
    :return: estimated t2, a list of population on the fitted curve
    """

    # define fitting function
    def fitRam(x, a, b):
        return - np.cos(a * x) * np.exp(- b * x) * 0.5 + 0.5

    # fit curve and obtain t2
    paraFit, _ = curve_fit(fitRam, tList, popList, [detuning, 0.])
    t2 = 1 / (paraFit[1] - 1 / (2 * t1))
    return t2, list(- np.cos(paraFit[0] * array(tList)) * np.exp(- paraFit[1] * array(tList)) * 0.5 + 0.5)


def qubitSpec(pulseModel: PulseModel, freqRange: list, sample: int, amp: float, t: float) -> [list, list]:
    """
    Qubit Spectroscopy. This function finds the qubit frequency by scanning the pulse frequency
    from a user-defined range.

    :param pulseModel: a pulseModel type object
    :param freqRange: a list of LO frequency's range
    :param sample: how many samples to scan within the freqRange
    :param amp: pulse amplitude (preferably Pi-pulse amp)
    :param t: pulse length (preferably Pi-pulse length)
    :return: a list of pulse frequency scanned and a list of population
    """

    # define pulse frequency list
    freqList = linspace(freqRange[0], freqRange[1], sample)

    # initialize job list
    QSpecJob = pulseModel.ham.createJobList()
    for freq in freqList:
        job = pulseModel.ham.createJob()
        job.setLO(driveX(3), 0, freq=freq)
        job.appendWave(driveX, 0, gaussian(t=t, a=amp, tau=t / 2, sigma=t / 8))
        QSpecJob.addJob(job)

    # run simulation and obtain populations from results
    stateInit = basis(3, 0)
    popList = []
    results = pulseModel.ham.simulate(jobList=QSpecJob, state0=stateInit)
    for result in results:
        rho = result['state']
        prj = basis(3, 1) @ dagger(basis(3, 1))
        popList.append(expect(prj, rho))
    return list(freqList), popList
