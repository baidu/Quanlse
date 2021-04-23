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


from typing import Dict, Union, Any, Callable, List, Tuple

import numpy
import math
from scipy.special import erf
import copy


def play(waveform: str, t: float, args: Dict[str, Union[numpy.ndarray, float]]) -> float:
    """
    Return the amplitude of given pulses.

    :param waveform: waveform
    :param t: time
    :param args: the standard control parameter dictionary (accroding to specific wave functions)
    :return: pulse amplitude value
    """
    if waveform == "quasi_square_erf":
        return quasiSquareErf(t, args)
    elif waveform == "gaussian":
        return gaussian(t, args)
    elif waveform == "square":
        return square(t, args)
    elif waveform == "sin":
        return sin(t, args)
    elif waveform == "drag_y1":
        return dragY1(t, args)
    elif waveform == "slepian":
        return slepian(t, args)
    else:
        assert False, "Unsupported wave function."


def quasiSquareErf(t: float, args: Dict[str, Union[numpy.ndarray, float]]) -> float:
    r"""
    Return the sample pulse with a quasi-square envelope. The specific wave form is expressed as:

    :math:`f(a, s, t_s, t_e) = \frac{a}{4} \left[ 1 + {\rm erf} \left( \sqrt{\pi} \frac{s}{a} (t-t_{s}) \right)
    \right] \times {\rm erfc} \left( \sqrt{\pi} \frac{s}{a} (t-t_{e}) \right)`

    :param t: time
    :param args: the standard control parameter dictionary: ``a``, ``s``, ``t_s``, ``t_e``
    :return: pulse amplitude value
    """
    a, idleLeft, idleRight = args["a"], args["idle_left"], args["idle_right"]
    if "sk" not in args.keys():
        sk = a * 0.3
    else:
        sk = args["sk"]
    if a == 0.0:
        return 0.0
    t1i = idleLeft
    t2i = idleRight

    pulse = 0.25 * a * (1 + erf(math.sqrt(math.pi) * sk / a * (t - t1i)))
    pulse = pulse * (1 - erf(math.sqrt(math.pi) * sk / a * (t - t2i)))
    return pulse


def gaussian(t: float, args: Dict[str, Union[numpy.ndarray, float]]) -> float:
    r"""
    Return the sample of pulse with a Gaussian envelope. The specific wave form is expressed as:

    :math:`\varepsilon(t) = a e^{-(t-\tau)^2/2\sigma^2}`

    :param t: time
    :param args: the standard control parameter dictionary: ``a``, ``tau``, ``sigma``
    :return: pulse amplitude value
    """
    a, tau, sigma = args["a"], args["tau"], args["sigma"]
    if sigma == 0:
        return 0
    pulse = a * math.exp(- ((t - tau) ** 2 / (2 * sigma ** 2)))
    return pulse


def square(t: float, args: Dict[str, Union[numpy.ndarray, float]]) -> float:
    r"""
    Return the sample pulse with a square envelope. The specific wave form is expressed as:

    :math:`\varepsilon(t) = a`

    :param t: time
    :param args: the standard control parameter dictionary: ``a``
    :return: pulse amplitude value
    """
    if args is None:
        return 0 * t
    return args["a"]


def sin(t: float, args: Dict[str, Union[numpy.ndarray, float]]) -> float:
    r"""
    Return the sample pulse with a sinusoidal envelope. The specific wave form is expressed as:

    :math:`\varepsilon(t) = a \sin(\omega t + \phi)`

    :param t: time
    :param args: the standard control parameter dictionary: ``a``, ``omega``, ``phi``
    :return: pulse amplitude value
    """
    a, omega, phi = args["a"], args["omega"], args["phi"]
    return a * math.sin(omega * t + phi)


def dragY1(t: float, args: Dict[str, Union[numpy.ndarray, float]]) -> float:
    r"""
    Return the sample pulse of Y-channel with DRAG technique. The specific wave form is expressed as:

    :math:`\varepsilon(t) = - a \frac{t - \tau}{\sigma^2} e^{-(t-\tau)^2/2\sigma^2}`

    :param t: time
    :param args: the standard control parameter dictionary: ``a``, ``tau``, ``sigma``
    :return: pulse amplitude value
    """
    a, tau, sigma = args["a"], args["tau"], args["sigma"]
    if sigma == 0:
        return 0
    pulse = - a * (t - tau) / (sigma ** 2) * math.exp(- ((t - tau) / sigma) ** 2 / 2)
    return pulse


def slepian(t: float, args: Dict[str, Union[numpy.ndarray, float]]) -> float:
    r"""
     Return the sample pulse of the Slepian function. The specific wave form is expressed as:

    :math:`E(t) = {t\over t_r} - \sum_i \frac{\lambda_i}{2\pi}\sin{\bigg(\frac{2\pi (i+1)}{t_r}t\bigg)}`

    :math:`\varepsilon(t) = A [E(t)-E(t-t_g+t_r)]`

    ``a`` is the amplitude :math:`A`; ``gate_time`` is the duration of the pulse time :math:`t_g`;
    ``rise_time`` is the duration of the rising edge :math:`t_r`; ``lambda_list`` is the list of :math:`\lambda_i`.

    :param t: time
    :param args: the standard control parameter dictionary: ``a``, ``lambda_list``, ``gate_time``, ``rise_time``
    :return: pulse value
    """

    if "lambda_list" not in args.keys():
        args["lambda_list"] = numpy.array([1.0280, -0.0606, 0.0052, 0.0055, 0.0055, 0.0047, 0.0046, 0.0035])

    def slepianStepAnalytic(tt):
        """ Obtain control pulse value using the slepian envelope """
        lambdaList = args['lambda_list']
        tr = args['rise_time']
        amp = args['a']

        n = len(lambdaList)
        sumReturn1 = 0.0
        sumReturn2 = 0.0
        for i in range(n):
            sumReturn1 = sumReturn1 + lambdaList[i] * (
                        1.0 * tt - (tr / (2.0 * numpy.pi * (i + 1))) * numpy.sin(2.0 * numpy.pi * (i + 1) * tt / tr))
            sumReturn2 = sumReturn2 + lambdaList[i] * tr
        funcVal = sumReturn1 * (numpy.heaviside(tt, 0) - numpy.heaviside(tt - tr, 0)) * amp / tr
        funcVal += sumReturn2 * numpy.heaviside(tt - tr, 0) * amp / tr
        return funcVal

    tf = args["gate_time"] - args["rise_time"]
    pulse = slepianStepAnalytic(t) - slepianStepAnalytic(t - tf)
    return pulse


def makeWaveData(ham: Dict[str, Any], name: str, t0: float, t: float = 0, f: Union[Callable, str] = None,
                 para: Dict[str, Any] = None, seq: List[float] = None) -> Dict[str, Any]:
    """
    Assemble a dictionary containing the details of a waveform.

    :param ham: the Hamiltonian dictionary
    :param name: the name of the control term
    :param t0: the start time of the pulse
    :param t: the duration of the pulse
    :param f: the function of wave function with the format of f(t, para)
    :param para: pulse parameters passed to ``f``
    :param seq: a list of pulse amplitudes
    """
    assert not (seq is not None and f is not None), "Cannot input seq and f at the same time."
    if f is None:
        assert seq is not None, "You should input one of func or seq."
        t = len(seq) * ham["circuit"]["dt"]
        # Record necessary information of the wave
        return {
            "name": name,
            "func": None,
            "para": para,
            "insert_ns": t0,
            "duration_ns": t,
            "sequence": seq
        }
    elif callable(f) or isinstance(f, str):
        if seq is not None:
            print("WARNING: func is given, hence the input of seq is ignored!")
        # Record necessary information of the wave
        return {
            "name": name,
            "func": f,
            "para": para,
            "insert_ns": t0,
            "duration_ns": t,
            "sequence": None
        }
    else:
        assert False, "Unsupported type of input for func, it should be a string, function or None."


def waveFuncToSeq(waveForm: Dict[str, Any], maxDt: int, dt: float) -> List[float]:
    """
    Transform the callable waveform to sequences.

    :param waveForm: waveData.
    :param maxDt: duration in unit dt (arbitrary wave generator (AWG) sampling time interval).
    :param dt: sampling time step.
    :return: a list of pulse sequence.
    """
    sequenceList = []
    # Traverse all the time slices.
    for nowDt in range(0, maxDt):
        nowNs = nowDt * dt + dt / 2
        insertNs = waveForm["insert_ns"]
        endNs = waveForm["insert_ns"] + waveForm["duration_ns"]
        if insertNs <= nowNs < endNs:
            # Calculate the waveforms' amplitudes.
            waveFunc = waveForm["func"]
            amp = waveFunc(nowNs - waveForm["insert_ns"], waveForm["para"])
        else:
            amp = 0.0
        sequenceList.append(amp)
    return sequenceList


def waveDataToSeq(data: Union[Dict[str, Any], List[Dict[str, Any]]], dt: float = 0.22, maxDt: int = None)\
        -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Transform the waveData to a JSON serializable datatype. This function transforms the callable
    waveform to sequences. If ``maxDt`` is not provided, this function will find the maxDt among
    the input waveData.

    :param data: waveData or waveData list
    :param dt: sampling time step
    :param maxDt: max time in unit dt
    :return: corresponding waveData
    """

    _data = copy.deepcopy(data)

    if maxDt is None:
        _, maxDt = computeWaveDataMaxTime(data, dt)

    if isinstance(data, list):
        for waveform in _data:
            if waveform["func"] is not None and callable(waveform["func"]):
                # print(f"Terms contains a callable waveform, it will be translated to a pulse sequence.")
                waveform["sequence"] = waveFuncToSeq(waveform, maxDt, dt)
                waveform["func"] = None
                waveform["para"] = None
    elif isinstance(data, dict):
        if _data["func"] is not None and callable(_data["func"]):
            # print(f"Terms contains a callable waveform, it will be translated to a pulse sequence.")
            _data["sequence"] = waveFuncToSeq(_data, maxDt, dt)
            _data["func"] = None
            _data["para"] = None
    else:
        assert False, "Only list or dictionary can be accepted by waveData."

    return _data


def computeWaveDataMaxTime(data: Union[Dict[str, Any], List[Dict[str, Any]]], dt: float) -> Tuple[float, int]:
    """
    Compute the pulse duration time of all the wave data input.

    :param data: waveData or waveData list
    :param dt: sampling time step
    :return: a tuple of duration time in Nano-second and dt (AWG sampling interval)
    """
    if isinstance(data, dict):
        maxNs = data["insert_ns"] + data["duration_ns"]
        maxDt = math.floor(maxNs / dt)
    else:
        # Find the max time
        maxNs = 0
        for waveform in data:
            finalNs = waveform["insert_ns"] + waveform["duration_ns"]
            if maxNs < finalNs:
                maxNs = finalNs
        maxDt = math.floor(maxNs / dt)

    return maxNs, maxDt


def addPulse(channel: str, t0: float = 0, t: float = 0, dt: float = 0.22,
             f: Union[Callable, str] = None, para: Dict[str, Any] = None,
             seq: List[float] = None) -> Dict[str, Any]:
    r"""
    Add pulses to x-y channel or readout channel.

    :param channel: the channel for x-y control or readout.
    :param t0: the start time of the pulse.
    :param t: the end time of the pulse.
    :param f: the callable function or the predefined string of the waveform.
    :param para: the parameters for waveform.
    :param seq: the sequence of the pulse.
    :param dt: the sampling period for each cycle.
    :return: the dictionary containing the information of the wave.
    """
    assert (channel == 'x' or channel == 'y' or channel == 'readout'),\
           'The name of the channel should be \'x\', \'y\' or \'readout\'.'

    assert not (seq is not None and f is not None),\
           'Cannot input seq and f at the same time.'

    if f is None:
        assert seq is not None, "You should input one of func or seq."

        t = len(seq) * dt

        # Record necessary information of the wave
        return {
            "name": channel,
            "func": None,
            "para": para,
            "insert_ns": t0,
            "duration_ns": t,
            "sequence": seq
        }
    elif callable(f) or isinstance(f, str):
        if seq is not None:
            print("WARNING: func is given, hence the input of seq is ignored!")

        # Record necessary information of the wave
        return {
            "name": channel,
            "func": f,
            "para": para,
            "insert_ns": t0,
            "duration_ns": t,
            "sequence": None
        }
    else:
        assert False,\
                'Unsupported type of input for func, it should be a string,\
                function or None.'
