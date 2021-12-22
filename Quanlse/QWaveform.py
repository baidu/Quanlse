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

r"""
In a superconducting quantum system, the control Hamiltonian represents external
control (microwave, magnetic flux etc.) acting on the qubits.

In Quanlse, users can use function ``addWave()`` or ``appendWave()`` to define and add wave to system.
Additionally, users can use ``clearWaves()`` to clear all waveforms in the specified control term.

Here, we will take ``addWave()`` or ``appendWave()`` to define and add the control wave. Each waveform
function ``A(t)`` can be defined using four arguments: start time ``t0``, duration ``t`` and the
corresponding parameters ``a``, ``tau``, ``sigma``. The function ``addWave()`` or ``appendWave()``
allows us to define waveforms in different ways:

**Example 1: using ``addWave()`` to add preset waveform functions**

Users can use the preset waveforms, here we use the Gaussian wave ``gaussian`` as an example:

.. code-block:: python

        ham.addWave(driveX, onSubSys=0, waves=gaussian(t=20, a=1.1, tau=10, sigma=4), t0=0)

**Example 2: using ``appendWave()`` to append preset waveform functions to the end of the sequence**

Users can use the preset waveforms, here we use the Gaussian wave ``gaussian`` as an example. The following
codes can append the waveform to the end of the existing sequence of the current control term `driveX`:

.. code-block:: python

        ham.appendWave(driveX, onSubSys=0, waves=gaussian(t=20, a=1.1, tau=10, sigma=4))

Users also can set ``compact`` parameter to ``False`` to append the waveform into the end of the whole sequence
of all control terms:

.. code-block:: python

        ham.appendWave(driveX, onSubSys=0, waves=gaussian(t=20, a=1.1, tau=10, sigma=4), compact=False)

**Example 3: using user-defined wave functions**

Users can also define a function in the form of ``func(_t, args)``, where the first parameter ``_t`` is the time
duration and ``args`` is the pulse parameters.

.. code-block:: python

        def userWaveform(t: Union[int, float], a: float, tau: float, sigma: float,
                         freq: float = None, phase: float = None) -> QWaveform:
            def func(_t, args):
                _a, _tau, _sigma = args
                if _sigma == 0:
                    return 0
                pulse = _a * exp(- ((_t - _tau) ** 2 / (2 * _sigma ** 2)))
                return pulse

            wave = QWaveform(f=func, t=t, args=(a, tau, sigma), freq=freq, phase=phase)
            wave.name = "user-defined wave"
            return wave
        ham.addWave(driveX, onSubSys=0, waves=userWaveform(t=20, a=1.1, tau=10, sigma=4), t0=0)


For more details, please visit https://quanlse.baidu.com/#/doc/tutorial-construct-ham.
"""

import copy
import json
import math
import pickle
import base64

from math import floor, pi
import matplotlib.pyplot as plt
from collections.abc import KeysView
from numpy import array, shape, linspace, arange
from typing import Dict, Any, List, Union, Tuple, Callable, Optional

from Quanlse.QPlatform import Error, Utilities
from Quanlse.QOperator import QOperator, driveX, driveY
from Quanlse.Utils.Plot import plotPulse
from Quanlse.Utils.WaveFunction import preset
from Quanlse.Utils.Functions import generateOperatorKey, combineOperatorAndOnSubSys, formatOperatorInput

QWAVEFORM_FLAG_ALWAYS_ON = 0b00001  # type: int
"""
QWaveform Flag: Users can input this flag to ``QJob.addWave()`` or ``QJob.appendWave()`` method to 
realize following function:

The waveform added will ignore the provided gate time, and adjust the gate time
according to the ``maxEndTime`` of the QJob.
"""

QWAVEFORM_FLAG_DO_NOT_CLEAR = 0b00010  # type: int
"""
QWaveform Flag: Users can input this flag to ``QJob.addWave()`` or ``QJob.appendWave()`` method to 
realize following function:

The waveform added will never be cleared by ``QJob.clearWaves()`` or ``QJob.appendWave()`` method,
and will be saved in a QHamiltonian instance, instead of a QJob instance.
Hence, when the QJob instance has no parent instance (QHamiltonian), no
QWAVEFORM_FLAG_DO_NOT_CLEAR waveform can be added.
"""

QWAVEFORM_FLAG_DO_NOT_PLOT = 0b00100  # type: int
"""
QWaveform Flag: Users can input this flag to ``QJob.addWave()`` or ``QJob.appendWave()`` method to 
realize following function:

The waveform added will be ignored by ``QJob.plot()``.
"""

QWAVEFORM_FLAG_PHASE_SHIFT = 0b01000  # type: int
"""
QWaveform Flag: Users can input this flag to ``QJob.addWave()`` or ``QJob.appendWave()`` method to 
realize following function:

The waveform cause some amount of phase shift. The amplitude will be ignore.
"""

QWAVEFORM_FLAG_MIX_WAVE = 0b10000  # type: int
"""
QWaveform Flag: Users can input this flag to ``QJob.addWave()`` or ``QJob.appendWave()`` method to 
realize following function:

This kind of waveform mix the two waves following the rules of microwave signal.
"""


class QWaveform:
    """
    Class of QWaveform.
    In Quanlse, we use a QWaveform object to store all information regarding a wave. A QWaveform object includes the
    wave's function (Gaussian, Square, QuasiSquare, etc.) and the respective parameters,
    a wave's starting time and duration, as well as the pulse frequency and phase, AWG sampling time, etc. QWaveform
    objects can be added to a control term in the Hamiltonian, giving a control term a specified waveform.

    :param f: a callable function, indicating a specific waveform
    :param t0: wave starting time
    :param t: wave duration
    :param args: wave arguments (takes different parameters for different waveform)
    :param seq: users can also input a python list consisting the wave sequence
    :param dt: AWG sampling time
    :param strength: wave strength
    :param freq: pulse frequency shift
    :param phase: pulse phase shift (will accumulate during the entire pulse execution)
    :param phase0: pulse phase shift (will not accumulate during the entire pulse execution)
    :param tag: wave tag indicating purpose
    :param flag: wave flag
    """

    def __init__(self, f: Union[Callable, str] = None, t0: Union[int, float] = 0.0, t: Union[int, float] = 0.0,
                 args: Any = None, seq: List[float] = None, dt: float = None, strength: float = 1.,
                 freq: float = None, phase: float = None, phase0: float = None, tag: str = None, flag: int = 0) -> None:
        """
        Constructor for QWaveform class.
        """

        if seq is not None and f is not None:
            raise Error.ArgumentError("Cannot input seq and f at the same time.")

        if f is None:
            if seq is None:
                raise Error.ArgumentError("You should input either func or seq.")
            # Record necessary information of the wave
            self.func = None  # type: Optional[Union[Callable, str]]
            self.t = None  # type: Optional[float]
            self.seq = copy.deepcopy(seq)  # type: Optional[List[Union[float, complex]]]
            self.args = None  # type: Optional[Any]
            self.name = ""  # type: Optional[str]
        elif callable(f) or isinstance(f, str):
            if seq is not None:
                print("WARNING: func is given, hence the input of seq is ignored!")
            # Record necessary information of the wave
            self.func = f  # type: Optional[Union[Callable, str]]
            self.t = t  # type: Optional[float]
            self.seq = None  # type: Optional[List[Union[float, complex]]]
            self.args = copy.deepcopy(args)  # type: Optional[Any]
            self.name = f if isinstance(f, str) else ""  # type: Optional[str]
        else:
            raise Error.ArgumentError("Unsupported type of input for func, it should either"
                                      " be a string, function or None.")

        # Common arguments.
        self.t0 = float(t0)  # type: float
        self.dt = None if dt is None else float(dt)  # type: Optional[float]
        self.strength = float(strength)  # type: float
        self.freq = freq  # type: Optional[float]
        self.phase = phase  # type: Optional[float]
        self.phase0 = phase0  # type: Optional[float]
        self.tag = tag  # type: Optional[str]
        self.flag = flag  # type: int

    def __str__(self) -> str:
        """
        Print description of the object.
        """
        returnStr = ""
        returnStr += f"Waveform: {self.name}\n"
        returnStr += f"    - tag: {self.tag}\n"
        returnStr += f"    - t0: {self.t0}\n"
        returnStr += f"    - t: {self.t}\n"
        returnStr += f"    - seq: {None if self.seq is None else len(self.seq)}\n"
        returnStr += f"    - args: {self.args}\n"
        returnStr += f"    - freq: {self.freq}\n"
        returnStr += f"    - phase: {self.phase}\n"
        returnStr += f"    - drive strength: {self.strength}\n"
        returnStr += f"    - flag: {self.flag}\n"

        return returnStr

    def __call__(self, *args, **kwargs):
        """
        Call self.func directly.
        """
        t = args[0]
        if self.func is not None:
            # Obtain the waveform
            if isinstance(self.func, str):
                # When self.func is a string
                func = preset(self.func)
            elif isinstance(self.func, Callable):
                # When self.func is a callable function
                func = self.func
            else:
                raise Error.ArgumentError(f"Unknown type ({type(self.func)}) of the input function!")
            # Run the function
            if len(args) == 1:
                return func(t, self.args)
            elif len(args) == 2:
                return func(t, args)
            elif len(args) == 0:
                raise Error.ArgumentError("You should input the time argument!")
            else:
                raise Error.ArgumentError(f"Too many arguments ({len(args)})!")
        elif self.seq is not None:
            if self.dt is None:
                raise Error.ArgumentError("Property dt is not set, cannot used to call a sequence-type QWaveform.")
            realTime = t
            if int(realTime / self.dt) >= len(self.seq):
                return 0.0
            else:
                return self.seq[int(realTime / self.dt)]
        else:
            raise Error.ArgumentError("Neither sequence nor func is set.")

    def waveFunctionToSequence(self, dt: float, maxEndTime: float = None) -> 'QWaveform':
        """
        Transform the function in QWaveform object to a sequence.

        :param dt: AWG sampling time
        :param maxEndTime: maximum ending time
        :return: returned QWaveform object
        """

        if self.func is not None:
            if dt is not None:
                _dt = dt
            elif self.dt is not None:
                _dt = self.dt
            else:
                raise Error.ArgumentError("Argument dt is not specified, please input dt.")
            seqList = []
            # QWAVEFORM_FLAG_ALWAYS_ON flag
            if self.flag & QWAVEFORM_FLAG_ALWAYS_ON:
                if maxEndTime is None:
                    print("Warning: QWaveform object with QWAVEFORM_FLAG_ALWAYS_ON flag"
                          " will become useless after serialization. You can input maxE"
                          "ndTime to serialize this object.")
                    endTimeDt = 0
                else:
                    endTimeDt = int(maxEndTime / _dt)
            else:
                endTimeDt = int(self.t / _dt)
            # Generate sequence
            for nowDt in range(endTimeDt):
                nowNs = nowDt * dt + _dt / 2
                seqList.append(self(nowNs))
            wave = sequence(t0=self.t0, seq=seqList, freq=self.freq, phase=self.phase)
            wave.tag = self.tag
            wave.flag = self.flag
            wave.dt = _dt
            return wave
        elif self.seq is not None:
            return copy.deepcopy(self)

    def plot(self, dt: float = None, xUnit: str = 'ns', yUnit: str = 'Amp (a.u.)',
             color: Union[str, List[str]] = None, dark: bool = False) -> None:
        """
        Plot the waveform.

        :param dt: AWG sampling time
        :param xUnit: X coordinate unit
        :param yUnit: Y coordinate unit
        :param color: pulse color options
        :param dark: whether activate dark mode
        :return: None
        """
        x = []
        y = []
        yLabel = []
        colors = []
        colorIndex = 0
        seqList = []
        _dt = None
        if self.func is not None:
            if dt is not None:
                _dt = dt
            elif self.dt is not None:
                _dt = self.dt
            elif self.seq is None:
                raise Error.ArgumentError("Argument dt is not specified, please input dt.")
            endTimeDt = int(self.t / _dt)
            for nowDt in range(endTimeDt):
                nowNs = nowDt * _dt + _dt / 2
                seqList.append(self(nowNs))
            tList = linspace(self.t0, self.t0 + self.t, len(seqList))
        else:
            seqList = self.seq
            tList = arange(0, len(seqList)).astype(int)

        y.append(list(seqList))
        x.append(list(tList))
        yLabel.append(yUnit)

        # Whether repetitive colors or all blue
        if color is None:
            colors.append('blue')
        else:
            colors.append(color[colorIndex])
        if self.name is not None:
            names = [self.name]
        else:
            names = ['']
        plotPulse(x, y, xLabel=f'Time ({xUnit})', yLabel=yLabel, title=names, color=colors, dark=dark)
        plt.show()

    def dump2Dict(self) -> Dict:
        """
        Return the dictionary which characterizes the current instance.
        """
        if isinstance(self.func, str):
            waveDict = {
                "func": "" if self.func is None else self.func,
                "args": {} if self.args is None else copy.deepcopy(self.args),
                "t": 0. if self.t is None else self.t,
                "t0": 0. if self.t0 is None else self.t0,
                "strength": 1. if self.strength is None else self.strength
            }
            if self.seq is not None:
                waveDict["seq"] = copy.deepcopy(self.seq)
            if self.name is not None:
                waveDict["name"] = self.name
            if self.seq is not None:
                waveDict["dt"] = self.dt
            if self.freq is not None:
                waveDict["freq"] = self.freq
            if self.phase is not None:
                waveDict["phase"] = self.phase
            if self.phase0 is not None:
                waveDict["phase0"] = self.phase0
            if self.tag is not None:
                waveDict["tag"] = self.tag
            if self.flag is not None:
                waveDict["flag"] = self.flag
            return waveDict
        else:
            raise Error.ArgumentError("Can not dump waveform with callable function to JSON string.")

    def dump2Json(self) -> str:
        """
        Return the JSON string which characterizes the current instance.
        """
        return json.dumps(self.dump2Dict())

    def dump(self, dt: float = None) -> str:
        """
        Return base64 encoded string.

        :param dt: AWG sampling time
        :return: returned base64 encoded string
        """
        obj = self.waveFunctionToSequence(dt)
        # For compatibility with previous versions
        if not hasattr(obj, "omega"):
            obj.omega = obj.freq
            obj.phi = 0.
        if not hasattr(obj, "phi"):
            obj.phi = obj.phase0
        byteStr = pickle.dumps(obj)
        base64str = base64.b64encode(byteStr)
        del obj
        return base64str.decode()

    def copy(self) -> 'QWaveform':
        """
        Return the copy of the object
        """
        return copy.deepcopy(self)

    @staticmethod
    def parseJson(jsonObj: Union[str, Dict]) -> 'QWaveform':
        """
        Create object from a JSON encoded string or dictionary.

        :param jsonObj: a JSON encoded string or dictionary
        :return: a QJob object
        """
        if isinstance(jsonObj, str):
            waveDict = json.loads(jsonObj)
        else:
            waveDict = copy.deepcopy(jsonObj)
        if "func" in waveDict.keys():
            waveObj = QWaveform(f=waveDict["func"])
        elif "seq" in waveDict.keys():
            waveObj = QWaveform(seq=waveDict["seq"])
        else:
            raise Error.ArgumentError("You should input either func or seq.")

        if "args" in waveDict.keys():
            waveObj.args = waveDict["args"]
        if "name" in waveDict.keys():
            waveObj.name = waveDict["name"]
        if "t" in waveDict.keys():
            waveObj.t = waveDict["t"]
        if "t0" in waveDict.keys():
            waveObj.t0 = waveDict["t0"]
        if "strength" in waveDict.keys():
            waveObj.strength = waveDict["strength"]
        if "dt" in waveDict.keys():
            waveObj.dt = waveDict["dt"]
        if "freq" in waveDict.keys():
            waveObj.freq = waveDict["freq"]
        if "phase" in waveDict.keys():
            waveObj.phase = waveDict["phase"]
        if "phase0" in waveDict.keys():
            waveObj.phase0 = waveDict["phase0"]
        if "tag" in waveDict.keys():
            waveObj.tag = waveDict["tag"]
        if "flag" in waveDict.keys():
            waveObj.flag = waveDict["flag"]

        return waveObj

    @staticmethod
    def load(base64Str: str) -> 'QWaveform':
        """
        Create object from base64 encoded string.

        :param base64Str: a input base64 encoded string
        :return: returned QWaveform object
        """
        byteStr = base64.b64decode(base64Str.encode())
        obj = pickle.loads(byteStr)  # type: QWaveform
        # For compatibility with previous versions
        if not hasattr(obj, "freq"):
            obj.freq = obj.omega if hasattr(obj, "omega") else None
        if not hasattr(obj, "phase0"):
            obj.phase0 = obj.phi if hasattr(obj, "phi") else None
        if not hasattr(obj, "phase"):
            obj.phase = None
        return obj


class QJob:
    """
    A QJob object in Quanlse contains all the information regarding a quantum task.
    The corresponding pulse information is stored in waveCache.

    :param subSysNum: subsystem's number
    :param sysLevel: system's energy level
    :param dt: AWG sampling time
    :param title: a user-given title
    :param description: a user-given description, preferably indicating the purpose
    :param ampGamma: An optional parameter indicating the amplitude's Gamma value
    """

    def __init__(self, subSysNum: int = None, sysLevel: Union[None, int, List[int]] = None, dt: float = None,
                 title: str = "", description: str = "", ampGamma: Optional[Union[int, float]] = None) -> None:
        """
        Constructor for QJob class.
        """

        self.title = title  # type: str
        self.description = description  # type: str
        self.dt = dt  # type: float
        self.subSysNum = subSysNum  # type: int
        self.sysLevel = sysLevel  # type: Union[int, List[int], None]
        self.tagWhiteListForClearWaves = []

        self.endTimeDt = 0  # type: int
        self.endTime = 0.0  # type: float

        self.ampGamma = ampGamma  # type: Optional[Union[int, float]]

        # Control operators
        self._ctrlOperators = {}  # type: Dict[str, Union[QOperator, List[QOperator]]]
        # List of QWaveform
        self._waves = {}  # type: Dict[str, Any]
        # List of local oscillator
        self._LO = {}  # type: Dict[str, Tuple[float, float]]
        # Sequence of cache for QWaveform
        self._waveCache = {}  # type: Dict[str, Any]
        # Parent object (QHamiltonian)
        self.parent = None  # type: 'QHamiltonian'

    def __len__(self):
        """
        Return the number of all waves from all operators.
        """
        counts = 0
        for key in self._waves.keys():
            counts += len(self._waves[key])
        return counts

    def __getitem__(self, key):
        """
        Return the Item.
        """
        return self._waves[key]

    def __add__(self, other):
        """
        Add two QJob Object, combine the operators with the waves into a new QJob obj.
        """
        newJob = copy.deepcopy(self)
        newJob.appendJob(other)
        return newJob

    def __str__(self) -> str:
        """
        Output the descriptions of the object.
        """
        returnStr = f"QJob ({'No title' if self.title == '' else self.title}):\n"
        for opKey in self._waves.keys():
            returnStr += f"- {opKey}  ({len(self._waves[opKey])} waveforms)\n"
            for wave in self._waves[opKey]:
                waveName = "No name" if wave.name is None or wave.name == '' else wave.name
                seqLen = f"{len(wave.seq)} pieces" if wave.seq is not None else 'None'
                returnStr += f"    - [{wave.tag}] {waveName}: t0={wave.t0}, t={wave.t}, args={wave.args}, " \
                             f"seq={seqLen}, strength={wave.strength}, freq={wave.freq}, " \
                             f"phase={wave.phase}.\n"

        return returnStr

    @property
    def subSysNum(self) -> int:
        """
        Return the number of the subsystems.
        """
        return self._subSysNum

    @subSysNum.setter
    def subSysNum(self, value: int):
        """
        Set the number of the subsystems.

        :param value: the number of the subsystems.
        """
        if not isinstance(value, int):
            raise Error.ArgumentError("subSysNum must be an integer!")
        if value < 1:
            raise Error.ArgumentError("subSysNum must be larger than 0!")
        self._subSysNum = value

    @property
    def sysLevel(self) -> Union[None, int, List[int]]:
        """
        Return the size of the subsystem.
        """
        return self._sysLevel

    @sysLevel.setter
    def sysLevel(self, value: Union[None, int, List[int]]):
        """
        Set the level of the subsystems.

        :param value: the number of the subsystems
        """
        if value is not None:
            if not isinstance(value, int) and not isinstance(value, list):
                raise Error.ArgumentError("sysSize must be an integer or a list!")
            if isinstance(value, list) and min(value) < 2:
                raise Error.ArgumentError("All items in sysSize must be larger than 1!")
            if isinstance(value, int) and value < 2:
                raise Error.ArgumentError(f"System level must be larger than 1, but {value} input!")
            self._sysLevel = value
        else:
            self._sysLevel = value

    @property
    def waveCache(self) -> Dict[str, Any]:
        """
        Return the cache of the pulses.
        """
        return self._waveCache

    @waveCache.deleter
    def waveCache(self):
        del self._waveCache

    @property
    def LO(self) -> Dict[str, Any]:
        """
        Return the cache of the pulses.
        """
        return self._LO

    @LO.setter
    def LO(self, value: Dict[str, Tuple[float, float]]):
        self._LO = copy.deepcopy(value)

    @property
    def waves(self) -> Dict[str, Union[QWaveform, List[QWaveform]]]:
        """
        Return the size of the subsystem.
        """
        return self._waves

    @property
    def ctrlOperators(self) -> Dict[str, Union[QOperator, List[QOperator]]]:
        """
        Return the dictionary of control operators.
        """
        return self._ctrlOperators

    @ctrlOperators.setter
    def ctrlOperators(self, value: Dict[str, Union[QOperator, List[QOperator]]]):
        """
        Set control operators.
        """
        # Check system information
        for opKey in value.keys():
            if isinstance(value[opKey], list):
                for op in value[opKey]:
                    if op.onSubSys + 1 > self._subSysNum:
                        raise Error.ArgumentError(f"Operator `{opKey}` is on the subsystem "
                                                  f"`{value[opKey].onSubSys}`, exceeds the s"
                                                  f"ubSysNum {self._subSysNum}.")

            elif value[opKey].onSubSys + 1 > self._subSysNum:
                raise Error.ArgumentError(f"Operator `{opKey}` is on the subsystem "
                                          f"`{value[opKey].onSubSys}`, exceeds the s"
                                          f"ubSysNum {self._subSysNum}.")

            if isinstance(self._sysLevel, int):
                if isinstance(value[opKey], list):
                    for op in value[opKey]:
                        opLvl = max(op.matrix.shape)
                        if opLvl != self._sysLevel:
                            raise Error.ArgumentError(f"Size of the operator `{opKey}` ({opLvl}) does not"
                                                      f"match with the system level ({self.sysLevel}).")
                else:
                    opLvl = max(value[opKey].matrix.shape)
                    if opLvl != self._sysLevel:
                        raise Error.ArgumentError(f"Size of the operator `{opKey}` ({opLvl}) does not"
                                                  f"match with the system level ({self.sysLevel}).")

            elif isinstance(self._sysLevel, list):
                if isinstance(value[opKey], list):
                    for op in value[opKey]:
                        opLvl = max(op.matrix.shape)
                        if opLvl != self._sysLevel[op.onSubSys]:
                            raise Error.ArgumentError(f"Size of the operator `{opKey}` ({opLvl}) does not"
                                                      f"match with the system level ({self.sysLevel}).")
                else:
                    opLvl = max(value[opKey].matrix.shape)
                    if opLvl != self._sysLevel[value[opKey].onSubSys]:
                        raise Error.ArgumentError(f"Size of the operator `{opKey}` ({opLvl}) on subsystem "
                                                  f"{value[opKey].onSubSys} does not match with the system "
                                                  f"level ({self.sysLevel}).")
        # Copy ctrl operators
        self._ctrlOperators = copy.deepcopy(value)
        # Initialize wave information
        for opKey in self._ctrlOperators.keys():
            if opKey not in self._waves:
                self._waves[opKey] = []

    @ctrlOperators.deleter
    def ctrlOperators(self):
        del self._ctrlOperators
        self._ctrlOperators = {}

    @property
    def dt(self) -> float:
        """
        Return the arbitrary wave generator's (AWG) sampling time interval (also stands for the step size for
        the piecewise-constant quantum simulation algorithm).
        """
        return self._dt

    @dt.setter
    def dt(self, value: float):
        """
        Set the arbitrary wave generator's (AWG) sampling time interval (also stands for the step size for
        the piecewise-constant quantum simulation algorithm).
        """
        if not isinstance(value, float):
            raise Error.ArgumentError("dt must be a float number!")
        if value < 0.0:
            raise Error.ArgumentError("dt must be larger than 0!")
        self._dt = value

    def appendWave(self, operators: Union[QOperator, Callable, List[QOperator], List[Callable]] = None,
                   onSubSys: Union[int, List[int]] = None, waves: Union[QWaveform, List[QWaveform]] = None,
                   strength: Union[int, float] = 1.0, freq: Optional[Union[int, float]] = None,
                   phase: Optional[Union[int, float]] = None, phase0: Optional[Union[int, float]] = None,
                   name: str = None, tag: str = None, shift: float = 0.0, compact: bool = True):
        """
        This method appends control terms and waveforms to a QJob object. Unlike `addWave()`, this function will append
        the waveform in the end of the existing waveforms, hence ignore the `t0` parameter of rhe `QWaveform` object.

        :param operators: wave operator
        :param onSubSys: what subsystem the wave is acting upon
        :param waves: a QWaveform object
        :param strength: wave strength
        :param freq: wave frequency shift
        :param phase: pulse phase shift (will accumulate during the entire pulse execution)
        :param phase0: pulse phase shift (will not accumulate during the entire pulse execution)
        :param name: a user-given name to the wave
        :param tag: wave tag indicating purpose
        :param shift: time shift of the added wave
        :param compact: the added wave will be left aligned in specified control terms if True.
        :return: None
        """
        # When searching for the waves, we do not filter the waves by the tags.
        wavesDict = self.searchWave(operators, onSubSys, names=name)
        maxTime = 0.
        if compact:
            # When compact is True, the wave will be appended to the final time of the existed waves from the
            #     specific control term.
            for waveKey in wavesDict.keys():
                if len(wavesDict[waveKey]) > 0:
                    for waveform in wavesDict[waveKey]:
                        if waveform.flag & QWAVEFORM_FLAG_ALWAYS_ON > 0:
                            continue
                        if waveform.seq is not None:
                            finalNs = waveform.t0 + len(waveform.seq) * self.dt
                        else:
                            finalNs = waveform.t0 + waveform.t
                        if maxTime < finalNs:
                            maxTime = finalNs
        else:
            # When compact is False, the wave will the appended to the final time of all existed waves of all
            #     control terms.
            maxTime, _ = self.computeMaxTime()

        # Finally, the start time of the appended wave is the maxTime plus the time shift value.
        t0 = maxTime + shift
        self.addWave(operators, onSubSys, waves, t0, strength, freq, phase, phase0, name, tag, None)

    def addWave(self, operators: Union[QOperator, Callable, List[QOperator], List[Callable]] = None,
                onSubSys: Union[int, List[int]] = None, waves: Union[QWaveform, List[QWaveform]] = None,
                t0: Union[int, float] = None, strength: Union[int, float] = 1.0,
                freq: Optional[Union[int, float]] = None, phase: Optional[Union[int, float]] = None,
                phase0: Optional[Union[int, float]] = None, name: str = None, tag: str = None, flag: int = None) \
            -> None:
        """
        This method adds control terms and waveforms to a QJob object. Users can specify the operators(QOperator object)
        , subsystem's indexes, wave information (QWaveform object), wave strength, wave frequency, wave phase. Moreover,
        the users can specify a tag and flag for identifying the purposes of the wave.

        :param operators: wave operator
        :param onSubSys: what subsystem the wave is acting upon
        :param waves: a QWaveform object
        :param t0: start time
        :param strength: wave strength
        :param freq: wave frequency shift
        :param phase: pulse phase shift (will accumulate during the entire pulse execution)
        :param phase0: pulse phase shift (will not accumulate during the entire pulse execution)
        :param name: give a new name for the input control operators
        :param tag: wave tag indicating purpose
        :param flag: wave flag
        :return: None
        """

        def _addOperator(_flag):
            """ Add Operator """
            if operators is not None and onSubSys is not None:
                # Generate the operator key
                _opKey = self.addOperators(operators, onSubSys, name=name,
                                           flagDoNotClear=(_flag & QWAVEFORM_FLAG_DO_NOT_CLEAR) > 0)
            elif operators is None and name is not None:
                _opKey = name
            else:
                raise Error.ArgumentError(
                    "You must indicate the operators with onSubSys or the name of the term.")
            return _opKey

        def _getFlagAndWaveContainer(_wave):
            """ Return Flag or wave container. """
            _realFlag = 0
            if flag is not None:
                _realFlag = flag
            elif _wave.flag is not None:
                _realFlag = _wave.flag
            if (_realFlag & QWAVEFORM_FLAG_DO_NOT_CLEAR) and self.parent is None:
                raise Error.ArgumentError("Current QJob object has no parent, hence cannot add "
                                          "waves with QWAVEFORM_FLAG_DO_NOT_CLEAR.")
            elif _realFlag & QWAVEFORM_FLAG_DO_NOT_CLEAR:
                _containerWaves = self.parent.doNotClearFlagWaves
            else:
                _containerWaves = self._waves
            return _realFlag, _containerWaves

        _waves = waves.copy()
        if isinstance(_waves, list):
            for wave in _waves:
                _flag, containerWaves = _getFlagAndWaveContainer(wave)
                opKey = _addOperator(_flag)
                wave.dt = self.dt
                wave.t0 = wave.t0 if t0 is None else t0
                wave.strength = wave.strength if strength is None else strength
                wave.freq = wave.freq if freq is None else freq
                wave.phase = wave.phase if phase is None else phase
                wave.phase0 = wave.phase0 if phase0 is None else phase0
                wave.tag = wave.tag if tag is None else tag
                wave.flag = _flag
                containerWaves[opKey].append(wave)
        elif isinstance(_waves, QWaveform):
            _flag, containerWaves = _getFlagAndWaveContainer(_waves)
            opKey = _addOperator(_flag)
            _waves.dt = self.dt
            _waves.t0 = _waves.t0 if t0 is None else t0
            _waves.strength = _waves.strength if strength is None else strength
            _waves.freq = _waves.freq if freq is None else freq
            _waves.phase = _waves.phase if phase is None else phase
            _waves.phase0 = _waves.phase0 if phase0 is None else phase0
            _waves.tag = _waves.tag if tag is None else tag
            _waves.flag = _flag
            containerWaves[opKey].append(_waves)
        else:
            raise Error.ArgumentError("Parameter waves must be a (or a list of) QWaveform object(s).")

        # Compute the end time
        self.computeMaxTime()
        self.clearCache()

    def addWaveRot(self, onSubSys: int, waves: Union[QWaveform, List[QWaveform]], t0: Union[int, float] = None,
                   detuning: Union[int, float] = 0.0, phase: Union[int, float] = 0.0, tag: str = None) -> None:
        """
        Add the control terms for the two-qubit cross-resonance effect in
        the rotating frame with the frequency of the diagonal elements of drift term.

        :param onSubSys: what subsystem the wave is acting upon
        :param waves: a QWaveform object
        :param t0: start time
        :param detuning: wave's corresponding detuning
        :param phase: wave phase
        :param tag: wave tag
        :return: None
        """
        levelList = []
        if isinstance(self.sysLevel, list):
            levelList = self.sysLevel
        elif isinstance(self.sysLevel, int):
            levelList = [self.sysLevel for _ in range(self.subSysNum)]

        self.addWave(operators=driveX(levelList[onSubSys]), onSubSys=onSubSys, t0=t0,
                     waves=copy.deepcopy(waves), freq=detuning, phase0=phase, tag=tag)
        self.addWave(operators=driveY(levelList[onSubSys]), onSubSys=onSubSys, t0=t0,
                     waves=copy.deepcopy(waves), freq=detuning, phase0=phase + pi / 2, tag=tag)

    def setLO(self, operators: Union[QOperator, Callable, List[QOperator], List[Callable]],
              onSubSys: Union[int, List[int]], name: str = None, freq: float = 1.0, phase: float = 0.0):
        """
        Add local oscillator for specific control terms.

        :param operators: QOperator object(s)
        :param onSubSys: what subsystem the wave is acting upon
        :param name: user-defined operator name
        :param freq: the frequency of the local oscillator
        :param phase: the phase between the local oscillator and the waveform
        """
        # Add the operator
        opKey = self.addOperators(operators, onSubSys, name)
        # Set the local oscillator frequency and the phase
        self._LO[opKey] = (freq, phase)

    def addOperators(self, operators: Union[QOperator, Callable, List[QOperator], List[Callable]] = None,
                     onSubSys: Union[int, List[int]] = None, name: str = None, flagDoNotClear: bool = False) -> str:
        """
        Add control terms onto the specified subsystem.

        :param operators: QOperator object(s)
        :param onSubSys: what subsystem the wave is acting upon
        :param name: user-defined operator name
        :param flagDoNotClear: whether this operator will be cleared
        """

        _operators = formatOperatorInput(operators, onSubSys, self.sysLevel)

        if _operators is not None and onSubSys is None:
            # Input the complete matrices directly.
            if isinstance(self._sysLevel, int):
                dim = self._sysLevel ** self._subSysNum
            elif isinstance(self._sysLevel, list):
                dim = 1
                for lvl in self._sysLevel:
                    dim = dim * lvl
            else:
                raise Error.ArgumentError(f"Unsupported type of self._sysLevel ({type(self._sysLevel)}).")

            if not shape(_operators.matrix) == (dim, dim):
                raise Error.ArgumentError(f"Dimension of operator ({dim}, {dim}) does not match the sysLevel.")

            operatorForSave = _operators
        elif _operators is not None and onSubSys is not None:
            # If users input the onSubSys, combineOperatorAndOnSubSys will update the onSubSys of
            #     the operators.
            operatorForSave = combineOperatorAndOnSubSys(self._subSysNum, _operators, onSubSys)
        elif _operators is None and name is not None:
            # If users input only name
            operatorForSave = None
        else:
            raise Error.ArgumentError("operators should not be None!")

        # Set the name if it is given
        if name is not None:
            opKey = name
        else:
            # Use function generateOperatorKey to generate the key of the operators.
            opKey = generateOperatorKey(self.subSysNum, operatorForSave)

        if flagDoNotClear and self.parent is None:
            raise Error.ArgumentError("Current QJob object has no parent, hence cannot add "
                                      "waves with QWAVEFORM_FLAG_DO_NOT_CLEAR.")
        elif not flagDoNotClear:
            containerCtrlOperators = self._ctrlOperators
            containerWaves = self._waves
        else:
            containerCtrlOperators = self.parent.doNotClearFlagOperators
            containerWaves = self.parent.doNotClearFlagWaves

        # Verify the uniqueness of the combination of the name and onSubSys
        if opKey not in containerCtrlOperators.keys():
            # Save the operators.
            containerCtrlOperators[opKey] = operatorForSave
            containerWaves[opKey] = []

        # Compute the end time
        self.computeMaxTime()

        return opKey

    def appendJob(self, obj: Any, t0: float = None) -> None:
        """
        Append objects to the QJob object.
        When timeShift is None, all waves appended will be started from the end of the current waves.

        :param obj: object to be added
        :param t0: start time
        :return: None
        """
        if isinstance(obj, QJob):
            # When the object is a QJob.
            if (obj.sysLevel is not None and self.sysLevel is not None) and obj.sysLevel != self.sysLevel:
                raise Error.ArgumentError("sysLevel does not match!")
            if obj.subSysNum != self.subSysNum:
                raise Error.ArgumentError("subSysNum does not match!")
            # Start adding
            self.computeMaxTime()
            if t0 is None:
                _timeShift = self.endTime
            else:
                _timeShift = t0
            _obj = copy.deepcopy(obj)
            for key in _obj.ctrlOperators.keys():
                if key not in self._ctrlOperators.keys():
                    # Current term exists, then add the wave directly.
                    self._ctrlOperators[key] = copy.deepcopy(_obj.ctrlOperators[key])
                    self._waves[key] = []
                # Then add the wave.
                for wave in _obj.waves[key]:
                    newWave = copy.deepcopy(wave)
                    newWave.t0 += _timeShift
                    self._waves[key].append(newWave)
            self.computeMaxTime()
        else:
            raise Error.ArgumentError(f"Unsupported type ({type(obj)}), only QJob is supported.")

    def clearWaves(self, operators: Union[QOperator, Callable, List[QOperator], List[Callable]] = None,
                   onSubSys: Union[int, List[int]] = None, names: Union[str, List[str]] = None,
                   tag: str = None) -> None:
        """
        Remove all waveforms in the specified control terms.
        If no names are given, remove all waveforms in all control terms.

        :param operators: the operator of the wave
        :param onSubSys: the subsystem of the wave
        :param names: the name of control term
        :param tag: the tag of the waves
        :return: None
        """
        # Generate opKeys of the waveforms which are going to be cleared
        if names is not None:
            # Set the name if it is given
            opKey = names
        elif operators is not None and onSubSys is not None:
            _operators = formatOperatorInput(operators, onSubSys, self.sysLevel)
            # If users input the onSubSys, combineOperatorAndOnSubSys will update the onSubSys of
            #     the operators.
            operatorForSave = combineOperatorAndOnSubSys(self._subSysNum, _operators, onSubSys)
            # Use function generateOperatorKey to generate the key of the operators.
            if isinstance(_operators, list):
                opKey = []
                for op in _operators:
                    opKey.append(generateOperatorKey(self.subSysNum, op))
            else:
                opKey = generateOperatorKey(self.subSysNum, operatorForSave)
        else:
            # Remove all waveforms
            opKey = self._ctrlOperators.keys()

        # Generate opKeys of the waveforms which are going to be cleared
        tagBlackList = []
        if tag is not None:
            if isinstance(tag, list):
                tagBlackList.extend(tag)
            else:
                tagBlackList.append(tag)

        # Generate new wave list
        newWaves = {}
        newCtrlOperators = {}
        for key in self._waves.keys():
            _toSave = []
            for wave in self._waves[key]:
                _ifSave = False
                # Check flag QWAVEFORM_DO_NOT_CLEAR
                _ifSave = _ifSave or ((wave.flag & QWAVEFORM_FLAG_DO_NOT_CLEAR) > 0)
                # Check opKey
                _ifSave = _ifSave or (key not in opKey)
                # Check tag
                _ifSave = _ifSave or (len(tagBlackList) > 0 and wave.tag not in tagBlackList)
                if _ifSave:
                    _toSave.append(copy.deepcopy(wave))
            if len(_toSave) > 0:
                newWaves[key] = _toSave
                newCtrlOperators[key] = copy.deepcopy(self._ctrlOperators[key])

        del self._waves
        del self._ctrlOperators
        self._waves = newWaves
        self._ctrlOperators = newCtrlOperators
        self.computeMaxTime()

    def searchWave(self, operators: Union[QOperator, Callable, List[QOperator], List[Callable]] = None,
                   onSubSys: Union[int, List[int]] = None, names: Union[str, List[str]] = None, tag: str = None):
        """
        Search waves by different methods.

        :param operators: search by operator
        :param onSubSys: search by subsystem
        :param names: search by name
        :param tag: search by tag
        """
        return self.searchWaveTool(self._waves, self._subSysNum, self._sysLevel, operators, onSubSys, names, tag)

    @staticmethod
    def searchWaveTool(container: Dict[str, List[QWaveform]], subSysNum: int, sysLevel: Union[int, List[int]],
                       operators: Union[QOperator, Callable, List[QOperator], List[Callable]] = None,
                       onSubSys: Union[int, List[int]] = None, names: Union[str, List[str]] = None,
                       tag: str = None) -> Dict[str, List[QWaveform]]:
        """
        Search waves by different methods.

        :param container: input dictionary
        :param subSysNum: subsystem number
        :param sysLevel: the system levels
        :param operators: search by operator
        :param onSubSys: search by subsystem
        :param names: search by name
        :param tag: search by tag
        """

        returnDict = {}
        if names is not None:
            def _searchWaveByNames(_returnDict, _name):
                """ Search Waves by name and tag """
                if _name in container.keys():
                    if tag is None:
                        _returnDict[_name] = container[_name]
                    else:
                        _returnDict[_name] = []
                        for _wave in container[_name]:
                            if _wave.tag == tag:
                                _returnDict[_name].append(_wave)

            # When names (of the operators) is not None, then find the waves by the name string
            if isinstance(names, str):
                # If user input a name
                _searchWaveByNames(returnDict, names)
                return returnDict
            elif isinstance(names, list):
                # If user input a list of names
                for name in names:
                    _searchWaveByNames(returnDict, name)
                return returnDict
            else:
                raise Error.ArgumentError("Variable names should be a list or int.")
        elif operators is not None and onSubSys is not None:
            # When names (of the operators) is None, then find the waves by the operators with onSubSys.
            formattedOperators = formatOperatorInput(operators, onSubSys, sysLevel)
            if isinstance(operators, list) and isinstance(onSubSys, list):
                _operators = formattedOperators
                _onSubSys = onSubSys
            else:
                _operators = [formattedOperators]
                _onSubSys = [onSubSys]

            if len(_operators) != len(_onSubSys):
                raise Error.ArgumentError("Length of operators and onSubSys must be the same.")

            for idx in range(len(_operators)):
                operatorForSearch = combineOperatorAndOnSubSys(subSysNum, _operators, _onSubSys)
                opKey = generateOperatorKey(subSysNum, operatorForSearch)
                if opKey in container.keys():
                    if tag is None:
                        returnDict[opKey] = container[opKey]
                    else:
                        returnDict[opKey] = []
                        for wave in container[opKey]:
                            if tag == wave.tag:
                                returnDict[opKey].append(wave)
            return returnDict
        elif tag is not None:
            for opKey in container.keys():
                returnDict[opKey] = []
                for waveform in container[opKey]:
                    if waveform.tag == tag:
                        returnDict[opKey].append(waveform)
            return returnDict
        else:
            raise Error.ArgumentError("You must indicate the operators with onSubSys / names / tag.")

    def plot(self, names: Union[str, List[str]] = None, tag: str = None, xUnit: str = 'ns', yUnit: str = 'Amp (a.u.)',
             color: Union[str, List[str]] = None, dark: bool = False) -> None:
        """
        Print the waveforms of the control terms listed in ``names``.

        :param names: the name or name list of the control term
        :param tag: the tag of the waveforms to be plotted.
        :param xUnit: the unit of the x axis
        :param yUnit: the unit of the y axis
        :param color: None or list of colors
        :param dark: the plot can be switched to dark mode if required by user
        :return: None

        In Quanlse, users can specify colors from:

        ``mint``, ``blue``, ``red``, ``green``, ``yellow``, ``black``, ``pink``, ``cyan``, ``purple``,
        ``darkred``, ``orange``, ``brown``, ``pink`` and ``teal``.

        The colors will repeat if there are more pulses than colors.
        """

        self.computeMaxTime()

        # Get the max time
        maxNs = self.endTime

        # print plot
        if names is None:
            nameList = []
            nameList.extend(self.waves.keys())
            if self.parent is not None:
                nameList.extend(self.parent.doNotClearFlagWaves.keys())
            names = sorted(nameList)
        elif isinstance(names, str):
            names = [names]
        elif isinstance(names, list):
            pass
        else:
            raise Error.ArgumentError("Variable names should be a list or str.")

        if len(names) < 1:
            raise Error.ArgumentError("No terms exist, thus no waves can be plotted.")

        # Keep track of figure numbers
        fig = 0

        # Create two empty lists
        x = []
        y = []
        yLabel = []
        colors = []
        colorIndex = 0
        counting = self.buildWaveCache(tag=tag, forPlot=True)
        # Plot waves
        plotNames = []
        isComplex = False
        for name in names:
            if name not in counting.keys() or counting[name] < 1:
                continue
            aList = array(self._waveCache[name])
            tList = linspace(0, maxNs, len(aList))
            y.append(list(aList))
            x.append(list(tList))
            yLabel.append(yUnit)
            plotNames.append(name)
            # If the amplitudes are complex number
            if aList.dtype in [complex]:
                isComplex = True
            # Whether repetitive colors or all blue
            if color is None:
                colors.append('blue')
            else:
                colors.append(color[colorIndex])
                colorIndex += 1
                if colorIndex == len(color):
                    colorIndex = 0
            fig += 1
        plotPulse(x, y, xLabel=f'Time ({xUnit})', yLabel=yLabel, title=plotNames,
                  color=colors, dark=dark, complx=isComplex)
        plt.show()
        self.clearCache()

    def plotIon(self, names: Union[str, List[str]] = None, xUnit: str = r'$\mu$s', yUnit: str = 'Amp (a.u.)',
                color: Union[str, List[str]] = None, dark: bool = False) -> None:
        """
        Print the waveforms of the control terms listed in ``names``.

        :param names: the name or name list of the control term
        :param xUnit: the unit of the x axis
        :param yUnit: the unit of the y axis
        :param color: None or list of colors
        :param dark: the plot can be switched to dark mode if required by user
        :return: None

        In Quanlse, users can specify colors from:

        ``mint``, ``blue``, ``red``, ``green``, ``yellow``, ``black``, ``pink``, ``cyan``, ``purple``,
        ``darkred``, ``orange``, ``brown``, ``pink`` and ``teal``.

        The colors will repeat if there are more pulses than colors.
        """

        self.computeMaxTime()

        # Get the max time
        maxNs = self.endTime

        # print plot
        if names is None:
            names = list(self.waves.keys())
        elif isinstance(names, str):
            names = [names]
        elif isinstance(names, list):
            pass
        else:
            raise Error.ArgumentError("Variable names should be a list or a str.")

        if len(names) < 1:
            raise Error.ArgumentError("No terms exist, hence no available waves can be plot.")

        # Keep track of figure numbers
        fig = 0

        # Create two empty lists
        x = []
        y = []
        yLabel = []
        colors = []
        colorIndex = 0
        self.buildWaveCache()

        for name in names:
            aList = array(self._waveCache[name])
            tList = linspace(0, maxNs, len(aList))
            y.append(list(aList))
            x.append(list(tList))
            yLabel.append(yUnit)

            # Whether repetitive colors or all blue
            if color is None:
                colors.append('blue')
            else:
                colors.append(color[colorIndex])
                colorIndex += 1
                if colorIndex == len(color):
                    colorIndex = 0
            fig += 1
        plotPulse(x, y, xLabel=f'Time ({xUnit})', yLabel=yLabel, title=names, color=colors, dark=dark)
        plt.show()
        self.clearCache()

    def keys(self) -> KeysView:
        """
        Return all the keys of waves.

        :return: returned keys
        """
        return self._waves.keys()

    def computeMaxTime(self) -> Tuple[float, float]:
        """
        Compute the time duration of the whole circuit according to the waves added.

        :return: returned total time duration Tuple[maxTime, maxDt]
        """
        # Find the longest time
        maxTime = 0.0
        containers = [self._waves]
        if self.parent is not None:
            containers.append(self.parent.doNotClearFlagWaves)
        for container in containers:
            for key in container.keys():
                waves = container[key]
                for waveform in waves:
                    if waveform.flag & QWAVEFORM_FLAG_ALWAYS_ON > 0:
                        continue
                    if waveform.seq is not None:
                        finalNs = waveform.t0 + len(waveform.seq) * self.dt
                    else:
                        finalNs = waveform.t0 + waveform.t
                    if maxTime < finalNs:
                        maxTime = finalNs
        maxDt = floor(maxTime / self.dt)

        self.endTimeDt = maxDt
        self.endTime = maxTime
        return maxTime, maxDt

    def clearCache(self) -> None:
        """
        Clear the wave cache.

        :return: None
        """
        self._waveCache = {}

    def buildWaveCache(self, tag: str = None, forPlot: bool = False) -> Dict[str, int]:
        """
        Generate the pulse sequences for further usage.

        :param tag: wave tag
        :param forPlot: whether plot
        :return: returned pulse sequences
        """
        # Generate the pulse sequences for all of the control terms.
        waveCounting = {}
        self.clearCache()
        allKeys = list(self.waves.keys())
        if self.parent is not None:
            allKeys.extend(self.parent.doNotClearFlagWaves.keys())
        for key in allKeys:
            _sequence = self.generatePulseSequence(name=key, tag=tag, forPlot=forPlot, counting=waveCounting)
            self._waveCache[key] = _sequence
        return waveCounting

    def generatePulseSequence(self, operators: Union[QOperator, Callable, List[QOperator], List[Callable]] = None,
                              onSubSys: Union[int, List[int]] = None, name: str = None, tag: str = None,
                              forPlot: bool = False, counting: Dict[str, int] = None,
                              phaseList: List[int] = None) -> List[Union[float, complex]]:
        """
        Generate the piecewise-constant pulse sequence according to the waveform configurations during
        the whole evolution time.

        :param operators: the list of operators
        :param onSubSys: the list of subsystems
        :param name: the key of control term
        :param tag: the tag of the waveforms
        :param forPlot: whether the sequence is for plotting
        :param counting: the waveform counting for all control terms
        :param phaseList: the list of phase on every time piece

        :return: the list of pulse sequence
        """

        _sequenceList = []
        _phaseList = []
        self.computeMaxTime()

        # Generate waveKey
        if name is not None:
            waveKey = name
        elif operators is not None and onSubSys is not None:
            _operators = formatOperatorInput(operators, onSubSys, self._sysLevel)
            operatorForSearch = combineOperatorAndOnSubSys(self._subSysNum, _operators, onSubSys)
            waveKey = generateOperatorKey(self.subSysNum, operatorForSearch)
        else:
            raise Error.ArgumentError("You must indicate the operators with onSubSys or the name of the term.")

        # Combine all waves (waves from current QJob instance and the parent QHamiltonian instance)
        combinedWaves = []
        if waveKey in self.waves:
            combinedWaves.extend(copy.deepcopy(self.waves[waveKey]))
        if self.parent is not None and waveKey in self.parent.doNotClearFlagWaves.keys():
            combinedWaves.extend(copy.deepcopy(self.parent.doNotClearFlagWaves[waveKey]))

        # A register for recording the phase accumulation for every control term.
        #     The values in this register may change over time.
        phaseAcc = {}
        phaseAccList = []

        # Traverse all the time slices.
        for nowDt in range(self.endTimeDt):
            # The amount of phase
            finalPhase = 0.
            # The amount of amplitude
            currentAmp = 0.
            # The nanosecond time of this moment
            nowNs = nowDt * self.dt + self.dt / 2.
            # The phase waveforms can work in this time window
            timeWindow = (nowNs - self.dt / 2., nowNs + self.dt / 2.)
            # Traverse all the waveforms
            for waveId, waveform in enumerate(combinedWaves):
                if waveform.flag & QWAVEFORM_FLAG_DO_NOT_PLOT > 0 and forPlot is True:
                    continue
                elif waveform.flag & QWAVEFORM_FLAG_ALWAYS_ON > 0:
                    waveform.t0 = 0.0
                    waveform.t = self.endTime
                else:
                    if tag is not None and waveform.tag != tag:
                        continue

                # Count the number of waveforms of the given operator
                if counting is not None:
                    if waveKey not in counting.keys():
                        counting[waveKey] = 0
                    counting[waveKey] += 1

                # Calculate the start and end time
                insertNs = waveform.t0
                if waveform.seq is not None:
                    endNs = waveform.t0 + len(waveform.seq) * self.dt
                else:
                    endNs = waveform.t0 + waveform.t

                # if the current waveform works
                if waveform.flag & QWAVEFORM_FLAG_PHASE_SHIFT > 0:
                    isWaveWorking = timeWindow[0] <= waveform.t0 < timeWindow[1]
                else:
                    isWaveWorking = insertNs <= nowNs < endNs

                if isWaveWorking:
                    # Calculate the waveforms' amplitudes.
                    waveform.dt = self.dt
                    if waveKey in self._LO.keys() or any([waveform.freq, waveform.phase0, waveform.phase]):
                        # The freq and phase shift set in QWaveform object
                        freq = 0. if waveform.freq is None else waveform.freq
                        phase = 0. if waveform.phase is None else waveform.phase
                        phase0 = 0. if waveform.phase0 is None else waveform.phase0
                        # Calculate the phase of the contemporary waveform
                        totalPhase = phase0
                        totalFreq = freq
                        # Plus the phase and frequency of the LO setup in QJob instance.
                        if waveKey in self._LO.keys():
                            totalFreq += self._LO[waveKey][0]
                            totalPhase += self._LO[waveKey][1]
                        # Calculate the accumulated phase in phaseAcc
                        if waveKey not in phaseAcc.keys():
                            phaseAcc[waveKey] = phase
                        else:
                            if waveId not in phaseAccList:
                                phaseAcc[waveKey] += phase
                                phaseAccList.append(waveId)
                        # The constant freq and phase set in QJob
                        totalPhase += phaseAcc[waveKey]
                        # Calculate the final LO value
                        if waveform.flag & QWAVEFORM_FLAG_MIX_WAVE > 0:
                            # If the waveform is the mixture of X and Y control
                            amp, varPhase = waveform(nowNs - waveform.t0)
                            finalPhase = totalPhase + varPhase
                            lo = math.cos(totalFreq * nowNs + finalPhase)
                            currentAmp += amp * lo
                        else:
                            finalPhase = totalPhase
                            lo = math.cos(totalFreq * nowNs + finalPhase)
                            currentAmp += waveform(nowNs - waveform.t0) * lo
                    else:
                        currentAmp += waveform(nowNs - waveform.t0)

                    # Modify the amplitudes by drive strength and amplitude noise.
                    currentAmp = currentAmp * waveform.strength
            _phaseList.append(finalPhase)
            _sequenceList.append(currentAmp)

        if phaseList is not None:
            phaseList = _phaseList
        del combinedWaves
        return _sequenceList

    def waveFunctionsToSequences(self, maxEndTime: float = None) -> 'QJob':
        """
        Convert all functions included QWaveform objects into a serializable sequence.

        :param maxEndTime: maximum ending time
        :return: returned QJob object
        """
        obj = copy.deepcopy(self)
        obj.clearWaves()
        for waveKey in self.waves.keys():
            # Add QOperator
            qOp = self.ctrlOperators[waveKey]
            if isinstance(qOp, list):
                onSubSys = []
                for op in qOp:
                    onSubSys.append(op.onSubSys)
            else:
                onSubSys = qOp.onSubSys
            obj.addOperators(qOp, onSubSys, waveKey)
            # Add QWaveform
            for wave in self.waves[waveKey]:
                seqWave = wave.waveFunctionToSequence(obj.dt, maxEndTime=maxEndTime)
                obj.addWave(name=waveKey, waves=seqWave, tag=wave.tag, strength=wave.strength, freq=wave.freq,
                            phase=wave.phase)
        return obj

    def dump(self, maxEndTime: float = None) -> str:
        """
        Return base64 encoded string.

        :param maxEndTime: maximum ending time
        :return: a base64 encoded string
        """
        # The critical step is transforming all waveform functions to the
        #     sequence, because functions can not the serialization.
        obj = self.waveFunctionsToSequences(maxEndTime=maxEndTime)
        obj.parent = None
        # For compatibility with previous versions
        for opKey in obj.waves.keys():
            for wave in obj.waves[opKey]:
                if not hasattr(wave, "omega"):
                    wave.omega = wave.freq
                    wave.phi = 0.
                if not hasattr(wave, "phi"):
                    wave.phi = 0. if wave.phase0 is None else wave.phase0
        # Dump the object
        byteStr = pickle.dumps(obj)
        base64str = base64.b64encode(byteStr)
        del obj
        return base64str.decode()

    def dump2Dict(self) -> Dict:
        """
        Return the dictionary which characterizes the current instance.
        """
        jobDict = {}
        # Basic information
        if self.subSysNum is not None:
            jobDict["subSysNum"] = self.subSysNum
        if self.sysLevel is not None:
            jobDict["sysLevel"] = self.sysLevel
        if self.title is not None:
            jobDict["title"] = self.title
        if self.description is not None:
            jobDict["description"] = self.description
        if self.ampGamma is not None:
            jobDict["ampGamma"] = self.ampGamma
        if self.dt is not None:
            jobDict["dt"] = self.dt
        if len(self.tagWhiteListForClearWaves) > 1:
            jobDict["tagWhiteListForClearWaves"] = copy.deepcopy(self.tagWhiteListForClearWaves)
        if len(self.LO) > 1:
            jobDict["LO"] = copy.deepcopy(self.LO)
        # Waveforms
        jobDict["waves"] = {}
        for _waveKey in self.waves:
            jobDict["waves"][_waveKey] = []
            for _wave in self.waves[_waveKey]:
                jobDict["waves"][_waveKey].append(copy.deepcopy(_wave.dump2Dict()))
        # Operators
        jobDict["operators"] = {}
        for _opKey in self.ctrlOperators:
            jobDict["operators"][_opKey] = []
            if isinstance(self.ctrlOperators[_opKey], list):
                for _subOp in self.ctrlOperators[_opKey]:
                    if _subOp.matrix is not None:
                        jobDict["operators"][_opKey].append({
                            "name": _subOp.name,
                            "onSubSys": _subOp.onSubSys,
                            "matrix": Utilities.numpyMatrixToDictMatrix(_subOp.matrix)
                        })
                    else:
                        jobDict["operators"][_opKey].append({
                            "name": _subOp.name,
                            "onSubSys": _subOp.onSubSys,
                        })
            else:
                _subOp = self.ctrlOperators[_opKey]
                if _subOp.matrix is not None:
                    jobDict["operators"][_opKey].append({
                        "name": _subOp.name,
                        "onSubSys": _subOp.onSubSys,
                        "matrix": Utilities.numpyMatrixToDictMatrix(_subOp.matrix)
                    })
                else:
                    jobDict["operators"][_opKey].append({
                        "name": _subOp.name,
                        "onSubSys": _subOp.onSubSys
                    })

        return jobDict

    def dump2Json(self):
        """
        Return the JSON string which characterizes the current instance.
        """
        return json.dumps(self.dump2Dict())

    def copy(self) -> 'QJob':
        """
        Return the copy of the object
        """
        return copy.deepcopy(self)

    @staticmethod
    def parseJson(jsonObj: Union[str, Dict]) -> 'QJob':
        """
        Create object from a JSON encoded string or dictionary.

        :param jsonObj: a JSON encoded string or dictionary
        :return: a QJob object
        """
        if isinstance(jsonObj, str):
            jsonDict = json.loads(jsonObj)
        else:
            jsonDict = copy.deepcopy(jsonObj)
        # Basic information
        subSysNum = jsonDict["subSysNum"] if "subSysNum" in jsonDict.keys() else None
        sysLevel = jsonDict["sysLevel"] if "sysLevel" in jsonDict.keys() else None
        title = jsonDict["title"] if "title" in jsonDict.keys() else ""
        description = jsonDict["description"] if "description" in jsonDict.keys() else ""
        dt = jsonDict["dt"] if "dt" in jsonDict.keys() else None
        ampGamma = jsonDict["ampGamma"] if "ampGamma" in jsonDict.keys() else None
        tagWhiteListForClearWaves = jsonDict["tagWhiteListForClearWaves"] if \
            "tagWhiteListForClearWaves" in jsonDict.keys() else []
        LO = jsonDict["LO"] if "LO" in jsonDict.keys() else {}
        # Create QJob instance
        job = QJob(subSysNum, sysLevel, dt, title, description, ampGamma)
        job.LO = LO
        job.tagWhiteListForClearWaves = tagWhiteListForClearWaves
        # Insert operators
        for _opKey in jsonDict["operators"]:
            _op = jsonDict["operators"][_opKey]
            if isinstance(_op, list):
                _opList = []
                _onSubSysList = []
                for _subOp in _op:
                    if "matrix" in _subOp.keys():
                        _matrix = Utilities.dictMatrixToNumpyMatrix(_subOp["matrix"], complex)
                    else:
                        _matrix = None
                    _opList.append(QOperator(name=_subOp["name"], matrix=_matrix))
                    _onSubSysList.append(_subOp["onSubSys"])
                job.addOperators(_opList, _onSubSysList, name=_opKey)
            else:
                if "matrix" in _op.keys():
                    _matrix = Utilities.dictMatrixToNumpyMatrix(_op["matrix"], complex)
                else:
                    _matrix = None
                job.addOperators(QOperator(name=_op["name"], matrix=_matrix), onSubSys=_op["onSubSys"], name=_opKey)
        # Insert waves
        for _waveKey in jsonDict["waves"]:
            for _waveDict in jsonDict["waves"][_waveKey]:
                _waveObj = QWaveform.parseJson(_waveDict)
                job.addWave(name=_waveKey, waves=_waveObj)
        return job

    @staticmethod
    def load(base64Str: str) -> 'QJob':
        """
        Create object from base64 encoded string.

        :param base64Str: a base64 encoded string
        :return: a QJob object
        """
        byteStr = base64.b64decode(base64Str.encode())
        obj = pickle.loads(byteStr)  # type: QJob

        # For compatibility with previous versions
        for opKey in obj.waves.keys():
            for wave in obj.waves[opKey]:
                if not hasattr(wave, "freq"):
                    wave.freq = wave.omega if hasattr(wave, "omega") else None
                if not hasattr(wave, "phase0"):
                    wave.phase0 = wave.phi if hasattr(wave, "phi") else None
                if not hasattr(wave, "phase"):
                    wave.phase = None

        return obj


class QJobList:
    """
    Class of QJobList

    :param subSysNum: subsystem number
    :param sysLevel: system energy level
    :param dt: AWG sampling time
    :param title: a user-given title
    :param description: a user-given description, preferably indicating the purpose
    """

    def __init__(self, subSysNum: int, sysLevel: Union[int, List[int]], dt: float, title: str = "",
                 description: str = "") -> None:
        """
        Constructor for QJobList class.
        """
        self._jobList = []  # type: List[QJob]
        self.title = title  # type: str
        self.description = description  # type: str
        self.subSysNum = subSysNum  # type: int
        self.sysLevel = sysLevel  # type: Union[int, List[int]]
        self._LO = {}  # type: Dict[str, Tuple[float, float]]
        self.dt = dt

    @property
    def subSysNum(self) -> int:
        """
        Return the number of the subsystems.
        """
        return self._subSysNum

    @subSysNum.setter
    def subSysNum(self, value: int):
        """
        Set the number of the subsystems.

        :param value: the number of the subsystems.
        """
        if not isinstance(value, int):
            raise Error.ArgumentError("subSysNum must be an integer!")
        if value < 1:
            raise Error.ArgumentError("subSysNum must be larger than 0!")
        self._subSysNum = value

    @property
    def sysLevel(self) -> Union[int, List[int]]:
        """
        Return the size of the subsystem.
        """
        return self._sysLevel

    @sysLevel.setter
    def sysLevel(self, value: Union[int, List[int]]):
        """
        Set the level of the subsystems.

        :param value: the number of the subsystems.
        """
        if not isinstance(value, int) and not isinstance(value, list):
            raise Error.ArgumentError("sysSize must be an integer or a list!")
        if isinstance(value, list) and min(value) < 2:
            raise Error.ArgumentError("All items in sysSize must be larger than 1!")
        if isinstance(value, int) and value < 2:
            raise Error.ArgumentError("All items in sysSize must be larger than 1!")
        self._sysLevel = value

    @property
    def LO(self) -> Dict[str, Any]:
        """
        Return the cache of the pulses.
        """
        return self._LO

    @LO.setter
    def LO(self, value: Dict[str, Tuple[float, float]]):
        self._LO = copy.deepcopy(value)

    @property
    def jobs(self) -> List[QJob]:
        return self._jobList

    def createJob(self) -> QJob:
        """
        Return a instance of QJob which has same system properties

        :return: returned QJob object
        """
        newJob = QJob(subSysNum=self.subSysNum, sysLevel=self.sysLevel, dt=self.dt)
        newJob.LO = self.LO
        return newJob

    def addJob(self, jobs: Union[QJob, List[QJob]]) -> None:
        """
        Add QJob(s) object into the job list.

        :param: jobs to be added
        :return: None
        """
        if isinstance(jobs, list):
            for job in jobs:
                if job.sysLevel is None and self.sysLevel is None and job.sysLevel != self.sysLevel:
                    raise Error.ArgumentError("sysLevel does not match!")
                if job.subSysNum != self.subSysNum:
                    raise Error.ArgumentError("subSysNum does not match!")
                self._jobList.append(copy.deepcopy(job))
        elif isinstance(jobs, QJob):
            if jobs.sysLevel is None and self.sysLevel is None and jobs.sysLevel != self.sysLevel:
                raise Error.ArgumentError("sysLevel does not match!")
            if jobs.subSysNum != self.subSysNum:
                raise Error.ArgumentError("subSysNum does not match!")
            self._jobList.append(copy.deepcopy(jobs))
        else:
            raise Error.ArgumentError("Unsupported input.")

    def clearJob(self) -> None:
        """
        Clear all QJob objects.

        :return: None
        """
        self._jobList = []

    def dump(self, maxEndTime: float = None) -> str:
        """
        Return base64 encoded string.

        :param maxEndTime: maximum ending time
        :return: a base64 encoded string
        """

        # The critical step is transforming all waveform functions to the
        #     sequence, because functions can not the serialization.
        obj = copy.deepcopy(self)
        obj.clearJob()
        for job in self._jobList:
            job.parent = None
            obj.addJob(job.waveFunctionsToSequences(maxEndTime=maxEndTime))

        # For compatibility with previous versions
        for job in obj.jobs:
            for opKey in job.waves.keys():
                for wave in job.waves[opKey]:
                    if not hasattr(wave, "omega"):
                        wave.omega = wave.freq
                        wave.phi = 0.
                    if not hasattr(wave, "phi"):
                        wave.phi = 0. if wave.phase0 is None else wave.phase0
        # Dump the object
        byteStr = pickle.dumps(obj)
        base64str = base64.b64encode(byteStr)
        del obj
        return base64str.decode()

    def clone(self) -> 'QJobList':
        """
        Return the copy of the object
        """
        return copy.deepcopy(self)

    @staticmethod
    def formatInputOperators() -> Union[List[QOperator], QOperator]:
        """
        The input operators may be the callables or the QOperator instances,
        """

    @staticmethod
    def load(base64Str: str) -> 'QJobList':
        """
        Create object from base64 encoded string.

        :param base64Str: a base64 encoded string
        :return: a QJobList object
        """
        byteStr = base64.b64decode(base64Str.encode())
        obj = pickle.loads(byteStr)  # type: QJobList
        # For compatibility with previous versions
        for job in obj.jobs:
            for opKey in job.waves.keys():
                for wave in job.waves[opKey]:
                    if not hasattr(wave, "freq"):
                        wave.freq = wave.omega if hasattr(wave, "omega") else None
                    if not hasattr(wave, "phase0"):
                        wave.phase0 = wave.phi if hasattr(wave, "phi") else None
                    if not hasattr(wave, "phase"):
                        wave.phase = None
        return obj


class QResult:
    """
    Class of QResult

    :param title: a user-given title
    :param description: a user-given description, preferably indicating the purpose
    """

    def __init__(self, title: str = "", description: str = "") -> None:
        """
        Constructor for class QResult
        """
        self.title = title
        self.description = description
        self._results = []

        # For iterator
        self._iter_count = 0

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, item: int) -> Any:
        return self._results[item]

    def __setitem__(self, key: int, value: Any):
        self._results[key] = value

    @property
    def result(self) -> Union[List, Dict]:
        return self._results

    def append(self, result: Any) -> None:
        """
        Set result.

        :param result: result to be set to
        :return: None
        """
        if isinstance(result, QResult):
            for item in result:
                self._results.append(item)
        else:
            self._results.append(result)

    def dump(self) -> str:
        """
        Return base64 encoded string.

        :return: a base64 encoded string
        """
        # Dump the object
        byteStr = pickle.dumps(self)
        base64str = base64.b64encode(byteStr)
        return base64str.decode()

    def clone(self) -> 'QResult':
        """
        Return the copy of the object
        """
        return copy.deepcopy(self)

    @staticmethod
    def load(base64Str: str) -> 'QResult':
        """
        Create object from base64 encoded string.

        :param base64Str: a base64 encoded string
        """
        byteStr = base64.b64decode(base64Str.encode())
        obj = pickle.loads(byteStr)  # type: QResult
        return obj


def gaussian(t: Union[float, List], a: float, tau: float, sigma: float, t0: Union[int, float] = 0.,
             freq: float = None, phase: float = None, phase0: float = None) -> QWaveform:
    """
    Return a QWaveform object of Gaussian wave.

    :param t: pulse duration
    :param a: pulse amplitude
    :param tau: pulse center position
    :param sigma: pulse standard deviation
    :param t0: start time
    :param freq: pulse frequency shift (will not accumulate during the entire pulse execution)
    :param phase: pulse phase shift (will accumulate during the entire pulse execution)
    :param phase0: pulse phase shift (will not accumulate during the entire pulse execution)
    :return: returned Gaussian QWaveform
    """
    args = {
        "a": a,
        "tau": tau,
        "sigma": sigma
    }
    wave = QWaveform(f="gaussian", t0=t0, t=t, args=args, freq=freq, phase=phase, phase0=phase0)
    wave.name = "gaussian"
    return wave


def square(t: Union[int, float], a: float, t0: Union[int, float] = 0., freq: float = None,
           phase: float = None, phase0: float = None) -> QWaveform:
    """
    Return a QWaveform object of square wave.

    :param t: pulse duration
    :param a: pulse amplitude
    :param t0: start time
    :param freq: pulse frequency shift (will not accumulate during the entire pulse execution)
    :param phase: pulse phase shift (will accumulate during the entire pulse execution)
    :param phase0: pulse phase shift (will not accumulate during the entire pulse execution)
    :return: returned square QWaveform
    """
    args = {
        "a": a
    }
    wave = QWaveform(f="square", t0=t0, t=t, args=args, freq=freq, phase=phase, phase0=phase0)
    wave.name = "square"
    return wave


def delay(t: Union[int, float], t0: Union[int, float] = 0., freq: float = None,
          phase: float = None, phase0: float = None) -> QWaveform:
    """
    Return a QWaveform object of delay.

    :param t: pulse duration
    :param t0: start time
    :param freq: pulse frequency shift (will not accumulate during the entire pulse execution)
    :param phase: pulse phase shift (will accumulate during the entire pulse execution)
    :param phase0: pulse phase shift (will not accumulate during the entire pulse execution)
    :return: returned square QWaveform
    """
    args = {
        "a": 0.
    }
    wave = QWaveform(f="square", t0=t0, t=t, args=args, freq=freq, phase=phase, phase0=phase0)
    wave.name = "delay"
    return wave


def virtualZ(phase: float = None, t0: Union[int, float] = 0.) -> QWaveform:
    """
    Return a QWaveform object of phase shift (implement Virtual-Z gate).

    :param phase: pulse phase shift (will accumulate during the entire pulse execution)
    :param t0: start time
    :return: returned square QWaveform
    """
    args = {
        "a": 0.
    }
    wave = QWaveform(f="square", t0=t0, t=0.0, args=args, freq=0.0, phase=phase, phase0=0.0,
                     flag=QWAVEFORM_FLAG_PHASE_SHIFT)
    wave.name = "virtualZ"
    return wave


def sin(t: Union[int, float], a: float, b: float, c: float, t0: Union[int, float] = 0.,
        freq: float = None, phase: float = None, phase0: float = None) -> QWaveform:
    """
    Return a QWaveform object of sin wave.
    x(t) = a * sin(b * t + c)

    :param t: pulse duration
    :param a: sin wave amplitude
    :param b: 2 * pi / (sin wave period)
    :param c: sin wave phase
    :param t0: start time
    :param freq: pulse frequency shift (will not accumulate during the entire pulse execution)
    :param phase: pulse phase shift (will accumulate during the entire pulse execution)
    :param phase0: pulse phase shift (will not accumulate during the entire pulse execution)
    :return: returned sin QWaveform
    """
    args = {
        "a": a,
        "b": b,
        "c": c
    }
    wave = QWaveform(f='sin', t0=t0, t=t, args=args, freq=freq, phase=phase, phase0=phase0)
    wave.name = "sin"
    return wave


def sequence(seq: List[Union[float, complex]], t0: Union[int, float] = 0., freq: float = None, phase: float = None,
             phase0: float = None) -> QWaveform:
    """
    Return a QWaveform object of pulse sequence.

    :param seq: pulse sequence
    :param t0: pulse start time
    :param freq: pulse frequency shift (will not accumulate during the entire pulse execution)
    :param phase: pulse phase shift (will accumulate during the entire pulse execution)
    :param phase0: pulse phase shift (will not accumulate during the entire pulse execution)
    :return: returned QWaveform object
    """
    wave = QWaveform(t0=t0, seq=seq, freq=freq, phase=phase, phase0=phase0)
    wave.name = "manual_sequence"
    return wave


def quasiSquareErf(t: Union[int, float], a: float, l: float, r: float, sk: float = None, t0: Union[int, float] = 0.,
                   freq: float = None, phase: float = None, phase0: float = None) -> QWaveform:
    """
    Return the sample pulse with a quasi-square envelope.

    :param t: pulse duration
    :param a: pulse amplitude
    :param l: quasiSquare wave function parameter
    :param r: quasiSquare wave function parameter
    :param sk: quasiSquare wave function parameter
    :param t0: pulse start time
    :param freq: pulse frequency shift shift (will not accumulate during the entire pulse execution)
    :param phase: pulse phase shift (will accumulate during the entire pulse execution)
    :param phase0: pulse phase shift (will not accumulate during the entire pulse execution)
    :return: returned quasi square QWaveform object
    """
    args = {
        "a": a,
        "l": l,
        "r": r,
        "sk": sk
    }
    wave = QWaveform(f="quasi_square_erf", t0=t0, t=t, args=args, freq=freq, phase=phase, phase0=phase0)
    wave.name = "quasi_square_erf"
    return wave


def dragY1(t: Union[int, float], a: float, tau: float, sigma: float, t0: Union[int, float] = 0.,
           freq: float = None, phase: float = None, phase0: float = None) -> QWaveform:
    """
    Return a QWaveform object of DRAG wave.

    :param t: pulse duration
    :param a: pulse amplitude
    :param tau: pulse center position
    :param sigma: pulse standard deviation
    :param t0: start time
    :param freq: pulse frequency shift (will not accumulate during the entire pulse execution)
    :param phase: pulse phase shift (will accumulate during the entire pulse execution)
    :param phase0: pulse phase shift (will not accumulate during the entire pulse execution)
    :return: returned QWaveform object with DRAG
    """
    args = {
        "a": a,
        "tau": tau,
        "sigma": sigma
    }
    wave = QWaveform(f="drag", t0=t0, t=t, args=args, freq=freq, phase=phase, phase0=phase0)
    wave.name = "drag"
    return wave


def mix(xWave: Optional[QWaveform], yWave: Optional[QWaveform], t0: Union[int, float] = 0.):
    """
    Return the mix of two QWaveform instances.

    :param xWave: the first QWaveform instance
    :param yWave: the second QWaveform instance
    :param t0:
    :return: returned mixed QWaveform
    """
    xDuration = 0. if xWave is None else xWave.t0 + xWave.t
    yDuration = 0. if yWave is None else yWave.t0 + yWave.t
    tDuration = max(xDuration, yDuration)
    args = {
        "xWave": xWave,
        "yWave": yWave,
    }
    wave = QWaveform(f="mix", t0=t0, t=tDuration, args=args, flag=QWAVEFORM_FLAG_MIX_WAVE)
    xName = "" if xWave is None else xWave.name
    yName = "" if yWave is None else yWave.name
    wave.name = "mix-{}-{}".format(xName, yName)
    return wave
