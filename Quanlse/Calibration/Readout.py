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
Readout resonator Calibration
"""

from Quanlse.Simulator import ReadoutPulse, ReadoutModel
from numpy import ndarray, argmin
from typing import Union, List, Dict, Tuple, Iterable
from math import pi
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def lorentzian(x: Union[int, float], x0: Union[int, float], a: Union[int, float],
               b: Union[int, float], c: Union[int, float]) -> Union[int, float]:
    """
    The lorentzian function.

    :param x: position.
    :param x0: The center position.
    :param a: The scale of the distribution.
    :param b: The width of the distribution.
    :param c: The shift of the amplitude.

    :return: the y value of the Lorentzian function
    """
    return (a / pi) * (0.5 * b) / ((x - x0) ** 2 + (0.5 * b) ** 2) + c


def fitLorentzian(x: ndarray, y: ndarray) -> Union[ndarray, Iterable, int, float]:
    """
    Fit the curve using Lorentzian function.

    :param x: a list of x data.
    :param y: a list of y data.
    :return: the result of curve fitting.
    """

    yMax = max(y)
    yMaxIdx = find_peaks(y, height=yMax)[0][0]
    yHalf = 0.5 * yMax

    yHalfIdx = argmin(abs(y - yHalf))

    freqCenter = x[yMaxIdx]
    width = 2 * (x[yMaxIdx] - x[yHalfIdx])

    param, cov = curve_fit(lorentzian, x, y, p0=[freqCenter, yMax, width, 0.])

    return param, cov


def resonatorSpec(readoutModel: ReadoutModel, onRes: List[int], freqRange: ndarray,
                  amplitude: Union[int, float], duration: Union[int, float],
                  qubitState='ground', loFreq: Union[int, float] = None) -> Tuple[Dict[int, List], Dict[int, List]]:
    """
    Resonator Spectroscopy.

    :param readoutModel: a ReadoutModel type object.
    :param onRes: Index of the resonators for simulation.
    :param freqRange: drive frequency's range.
    :param amplitude: amplitude of the readout pulse.
    :param duration: duration of the readout pulse.
    :param qubitState: the initial qubit state.
    :param loFreq: lo frequency for demodulation.

    :return: a tuple of vi data and vq data.
    """

    # Initialize readoutPulse object
    driveStrength = {}

    for idx in onRes:
        driveStrength[idx] = amplitude

    if loFreq is not None:
        loFreq = loFreq
    else:
        # The local oscillator frequency for demodulation.
        loFreq = 6.064 * (2 * pi)

    readoutPulse = ReadoutPulse(driveStrength=driveStrength, driveFreq={}, loFreq=loFreq)

    readoutModel.readoutPulse = readoutPulse

    viDict = {}
    vqDict = {}

    for idx in onRes:
        viDict[idx] = []
        vqDict[idx] = []

    for freq in freqRange:

        for idx in onRes:
            readoutModel.readoutPulse.setFreq(idx, freq)

        result = readoutModel.simulate(duration=duration, resIdx=onRes, state=qubitState)

        vi = result['vi']
        vq = result['vq']

        for idx in onRes:
            viDict[idx].append(vi[idx])
            vqDict[idx].append(vq[idx])

    return viDict, vqDict


def findFreq(y: Iterable) -> Tuple[ndarray, dict]:
    """
    Find the index of the peak.

    :param y: a list of signals.
    :return: the index of the peak.
    """

    yMax = max(y)
    yHalf = yMax / 2

    idx = find_peaks(y, height=yHalf)

    return idx
