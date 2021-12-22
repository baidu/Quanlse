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
Define the preset waveform functions.
"""

from typing import List, Any, Callable
from math import floor, sqrt, pi, atan, sin
from scipy.special import erf
from numpy import array, shape, linspace, arange, exp, sign

from Quanlse.QPlatform.Error import ArgumentError


def waveGaussian(t, args):
    """ The Gaussian function """
    a, tau, sigma = args["a"], args["tau"], args["sigma"]
    if sigma == 0:
        return 0
    pulse = a * exp(- ((t - tau) ** 2 / (2 * sigma ** 2)))
    return pulse


def waveSquare(t, args):
    """ The constant function """
    return args["a"]


def waveSin(t, args):
    """ The sin function """
    a, b, c = args["a"], args["b"], args["c"]
    return a * sin(b * t + c)


def waveQuasiSquareErf(t, args):
    """ The quasi square error function """
    a, l, r, sk = args["a"], args["l"], args["r"], args["sk"]
    if sk is None:
        sk = a * 0.3
    if a == 0.0:
        return 0.0
    t1i = l
    t2i = r
    pulse = 0.25 * a * (1 + erf(sqrt(pi) * sk / a * (t - t1i)))
    pulse = pulse * (1 - erf(sqrt(pi) * sk / a * (t - t2i)))
    return pulse


def waveDrag(t, args):
    """ The DRAG function """
    a, tau, sigma = args["a"], args["tau"], args["sigma"]
    if sigma == 0:
        return 0
    pulse = - a * (t - tau) / (sigma ** 2) * exp(- ((t - tau) / sigma) ** 2 / 2)
    return pulse


def waveMix(t, args):
    """ The mix function """
    xWave, yWave = args["xWave"], args["yWave"]
    xVal = 0. if xWave is None else xWave(t)
    yVal = 0. if yWave is None else yWave(t)
    # Calculate amplitude
    amp = sqrt(xVal ** 2 + yVal ** 2)
    # Calculate phase
    if abs(xVal) > 1e-10 and abs(yVal) > 1e-10:
        varPhase = atan(yVal / xVal)
    elif abs(xVal) < 1e-10 < abs(yVal):
        varPhase = sign(yVal) * pi / 2
    else:
        varPhase = 0.
    return amp, varPhase


presetList = {
    "gaussian": {
        "func": waveGaussian,
        "description": "The gaussian wave function: x(t) = a * exp(- ((t - tau) ** 2 / (2 * sigma ** 2))).",
        "args": {
            "a": "Maximum amplitude: float",
            "tau": "The center position: float",
            "sigma": "The standard deviation: float"
        }
    },
    "square": {
        "func": waveSquare,
        "description": "The square wave function: x(t) = a.",
        "args": {
            "a": "Maximum amplitude: float"
        }
    },
    "sin": {
        "func": waveSin,
        "description": "The sin wave function: x(t) = a * sin(b * t + c).",
        "args": {
            "a": "Sin wave amplitude: float",
            "b": "2 * pi / (sin wave period): float",
            "c": "Sin wave phase: float",
        }
    },
    "quasi_square_erf": {
        "func": waveQuasiSquareErf,
        "description": "The quasi-square envelope.",
        "args": {
            "a": "pulse amplitude: float",
            "l": "quasiSquare wave function parameter: float",
            "r": "quasiSquare wave function parameter: float",
            "sk": "quasiSquare wave function parameter: float",
        }
    },
    "drag": {
        "func": waveDrag,
        "description": "The DRAG (Derivative Removal by Adiabatic Gate) waveform.",
        "args": {
            "a": "Maximum amplitude: float",
            "tau": "The center position: float",
            "sigma": "The standard deviation: float"
        }
    },
    "mix": {
        "func": waveMix,
        "description": "Mix two waveforms.",
        "args": {
            "xWave": "The first QWaveform instance: QWaveform",
            "yWave": "The second QWaveform instance: QWaveform",
        }
    }
}
"""
The preset wave function dictionary.
"""


def preset(func: str) -> Callable:
    """
    Return the preset waveform function according to the string function name.

    :param func: the name of the function
    """
    if func in presetList.keys():
        return presetList[func]["func"]
    else:
        raise ArgumentError(f"Function {func} does not exist.")

