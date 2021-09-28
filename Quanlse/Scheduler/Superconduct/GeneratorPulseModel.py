#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2020 Baidu, Inc. All Rights Reserved.
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
Pulse Generator for PulseModel.
"""

from numpy import pi, cos, sin

from Quanlse.QPlatform.Error import ArgumentError
from Quanlse.QHamiltonian import QHamiltonian
from Quanlse.QOperator import uWave, flux, driveX, driveY, driveZ
from Quanlse.QWaveform import QJob, square, gaussian, dragY1, quasiSquareErf, virtualZ, mix
from Quanlse.Scheduler.Superconduct import SchedulerPulseGenerator
from Quanlse.QOperation.RotationGate import RX, RY, RZ
from Quanlse.QOperation.FixedGate import H, CZ
from Quanlse.QOperation import CircuitLine


def pulseGenerator(ham: QHamiltonian) -> SchedulerPulseGenerator:
    r"""
    The pulseGenerator for the simulator.

    :param ham: a QHamiltonian object.
    :return: a SchedulerPulseGenerator object.
    """
    generator = SchedulerPulseGenerator(ham)

    # Add the basic generator for single-qubit gates
    def generateBasic1Q(ham, gate, onQubit, scheduler) -> QJob:
        """ Generate the single-qubit gates """
        job = ham.createJob()

        # Generate the operator
        if gate.name in ['X', 'RX', 'Y', 'RY']:
            if "caliDataXY" not in scheduler.conf.keys():
                raise ArgumentError(f"'caliDataXY' does not exist in scheduler.conf, "
                                    f"hence X/Y controls are not supported.")
            if onQubit not in scheduler.conf["caliDataXY"].keys():
                raise ArgumentError(f"Configuration of qubit(s) {onQubit} does not, "
                                    f"hence X/Y controls are not supported.")
            # Pulse Parameters
            piLen = scheduler.conf["caliDataXY"][onQubit]["piLen"]
            piAmp = scheduler.conf["caliDataXY"][onQubit]["piAmp"]
            piTau = scheduler.conf["caliDataXY"][onQubit]["piTau"]
            piSigma = scheduler.conf["caliDataXY"][onQubit]["piSigma"]
            dragCoef = scheduler.conf["caliDataXY"][onQubit]["dragCoef"]
            if gate.name in ['X', 'RX']:
                amp = piAmp if gate.name == 'X' else gate.uGateArgumentList[0] / pi * piAmp
                piShift = 0.
            elif gate.name in ['Y', 'RY']:
                amp = piAmp if gate.name == 'Y' else gate.uGateArgumentList[0] / pi * piAmp
                piShift = pi / 2
            else:
                raise ArgumentError(f"Unsupported gate {gate.name}!")

            # Add the waveform to job
            _frameMode = scheduler.conf["frameMode"]
            if _frameMode == 'rot':
                job.addWave(driveX, onQubit, gaussian(piLen, amp * cos(piShift), piTau, piSigma), t0=0.)
                job.addWave(driveX, onQubit, dragY1(piLen, dragCoef * amp * sin(- piShift), piTau, piSigma), t0=0.)
                job.addWave(driveY, onQubit, gaussian(piLen, amp * sin(piShift), piTau, piSigma), t0=0.)
                job.addWave(driveY, onQubit, dragY1(piLen, dragCoef * amp * cos(piShift), piTau, piSigma), t0=0.)
            elif _frameMode == 'lab':
                gaussianWave = gaussian(piLen, amp, piTau, piSigma, phase0=piShift)
                dragWave = dragY1(piLen, dragCoef * amp, piTau, piSigma, phase0=piShift)
                job.appendWave(uWave, onQubit, mix(gaussianWave, dragWave))
            else:
                raise ArgumentError(f"Unsupported frameMode '{_frameMode}'!")
        elif gate.name in ['Z', 'RZ']:
            if "caliDataZ" not in scheduler.conf.keys():
                raise ArgumentError(f"'caliDataZ' does not exist in scheduler.conf, "
                                    f"hence Z controls are not supported.")
            if onQubit not in scheduler.conf["caliDataZ"].keys():
                raise ArgumentError(f"Configuration of qubit(s) {onQubit} does not, "
                                    f"hence Z controls are not supported.")
            # Pulse Parameters
            piLen = scheduler.conf["caliDataZ"][onQubit]["piLen"]
            piAmp = scheduler.conf["caliDataZ"][onQubit]["piAmp"]
            amp = piAmp if gate.name == 'Z' else gate.uGateArgumentList[2] / pi * piAmp

            # Add the waveform to job
            job.appendWave(flux, onQubit, square(piLen, amp))

        else:
            raise ArgumentError(f"Unsupported gate {gate.name}!")

        return job

    # Add the generator for single-qubit gates
    def generate1Q(ham, cirLine, scheduler) -> QJob:
        onQubit = int(cirLine.qRegIndexList[0])
        if cirLine.data.name in ['X', 'RX', 'Y', 'RY', 'Z', 'RZ']:
            return generateBasic1Q(ham, cirLine.data, onQubit, scheduler)
        elif cirLine.data.name == 'H':
            jobX = generateBasic1Q(ham, RX(pi), onQubit, scheduler)
            jobY = generateBasic1Q(ham, RY(-pi / 2), onQubit, scheduler)
            return jobX + jobY
        else:
            raise ArgumentError(f"Unsupported gate {cirLine.data.name}!")

    # Add the generator for two-qubit gates
    def generateCz(ham, cirLine, scheduler) -> QJob:
        """ Generate the CZ gate """
        job = ham.createJob()
        qIndex = list(cirLine.qRegIndexList)
        if tuple(qIndex) not in scheduler.conf["caliDataCZ"].keys():
            qIndex.sort()
        qIndex = tuple(qIndex)
        # Pulse parameters
        _frameMode = scheduler.conf["frameMode"]
        czLen = scheduler.conf["caliDataCZ"][qIndex]["czLen"]
        q0ZAmp = scheduler.conf["caliDataCZ"][qIndex]["q0ZAmp"]
        q1ZAmp = scheduler.conf["caliDataCZ"][qIndex]["q1ZAmp"]
        q0VZPhase = scheduler.conf["caliDataCZ"][qIndex]["q0VZPhase"]
        q1VZPhase = scheduler.conf["caliDataCZ"][qIndex]["q1VZPhase"]
        q0ZWave = quasiSquareErf(czLen, q0ZAmp, 0.1 * czLen, 0.9 * czLen, 0.25 * q0ZAmp)
        q1ZWave = quasiSquareErf(czLen, q1ZAmp, 0.1 * czLen, 0.9 * czLen, 0.25 * q1ZAmp)
        if _frameMode == 'lab':
            # Add the square wave
            job.addWave(flux, qIndex[0], q0ZWave, t0=0.)
            job.addWave(flux, qIndex[1], q1ZWave, t0=0.)
            # Add the virtual Z gate phase
            job.addWave(uWave, qIndex[0], virtualZ(q0VZPhase), t0=0.)
            job.addWave(uWave, qIndex[1], virtualZ(q1VZPhase), t0=0.)
        elif _frameMode == 'rot':
            # Add the square wave
            job.addWave(driveZ, qIndex[0], q0ZWave, t0=0.)
            job.addWave(driveZ, qIndex[1], q1ZWave, t0=0.)
            # Add the virtual Z gate phase
            job.addWave(driveX, qIndex[0], virtualZ(q0VZPhase), t0=0.)
            job.addWave(driveY, qIndex[0], virtualZ(q0VZPhase), t0=0.)
            job.addWave(driveX, qIndex[1], virtualZ(q1VZPhase), t0=0.)
            job.addWave(driveY, qIndex[1], virtualZ(q1VZPhase), t0=0.)
        else:
            raise ArgumentError(f"Unsupported frameMode '{_frameMode}'!")
        return job

    def generateCNOT(ham, cirLine, scheduler) -> QJob:
        """ Generate the CNOT gate """
        qIndex = list(cirLine.qRegIndexList)
        jobH1 = generate1Q(ham, CircuitLine(H, [qIndex[1]]), scheduler)
        jobCZ = generateCz(ham, CircuitLine(CZ, qIndex), scheduler)
        return jobH1 + jobCZ + jobH1

    # Add the Generator methods to the generator instance
    generator.addGenerator(['X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'H', 'U'], generate1Q)
    generator.addGenerator(['CZ'], generateCz)
    generator.addGenerator(['CNOT'], generateCNOT)

    return generator
