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
Qubit-Coupler-Qubit architecture template and analysis toolkit
"""

import copy
from Quanlse.QPlatform.Error import ArgumentError
from typing import List, Union, Dict, Optional
from numpy import pi, ndarray, identity, trace, diag, array
from Quanlse.Simulator import PulseModel
from Quanlse.QHamiltonian import QHamiltonian
from Quanlse.QPlatform import Error
from Quanlse.QWaveform import gaussian, QJob, quasiSquareErf, virtualZ
from Quanlse.QOperator import create, destroy, dagger, sigmaZ, sigmaX, sigmaY, uWave
from Quanlse.Utils.Functions import tensor, basis, findIndex, blockDiag, eigenSystem
from Quanlse.Scheduler.Superconduct import Scheduler, SchedulerPulseGenerator
from Quanlse.QOperation import CircuitLine
from Quanlse.QOperation.FixedGate import CR, X, EchoedCR
from Quanlse.QOperation.RotationGate import VZ, RX, RY


def pulseGenerator(ham: QHamiltonian) -> SchedulerPulseGenerator:
    r"""
    The pulseGenerator for the simulator.

    :param ham: a QHamiltonian object.
    :return: a SchedulerPulseGenerator object.
    """
    generator = SchedulerPulseGenerator(ham)

    def generateVZ(ham: QHamiltonian, cirLine: CircuitLine) -> QJob:
        """
        Generate the virtual-z gate
        """
        job = ham.createJob()

        onQubit = int(cirLine.qRegIndexList[0])

        phase = cirLine.data.uGateArgumentList[2]

        vz = virtualZ(phase=phase)

        job.appendWave(operators=uWave, onSubSys=onQubit, waves=vz)

        return job

    def generateHadamard(ham: QHamiltonian, cirLine: CircuitLine, scheduler: Scheduler) -> QJob:
        """
        Generate Hadamard gate.
        """

        qIndex = list(cirLine.qRegIndexList)
        jobRX = generate1QGate(ham, CircuitLine(RX(pi), qIndex), scheduler)
        jobRY = generate1QGate(ham, CircuitLine(RY(-pi / 2), qIndex), scheduler)

        return jobRX + jobRY

    def generateCR(ham: QHamiltonian, cirLine, scheduler, ampFactor=1., tFactor=1.) -> QJob:
        """
        Generate the CR gate
        :ham: hamiltonian.
        :cirLine: circuitLine.
        :scheduler: scheduler.
        :ampFactor: the scale of the amplitude.
        :tFactor: the scale of the time gate.
        """

        qIndex = list(cirLine.qRegIndexList)
        if tuple(qIndex) not in scheduler.conf["caliDataCR"].keys():
            qIndex.sort()
        qIndex = tuple(qIndex)

        # Pulse parameters
        CRtg = tFactor * scheduler.conf["caliDataCR"][qIndex]["CRtg"]
        CRamp = ampFactor * scheduler.conf["caliDataCR"][qIndex]["CRamp"]
        CRfreq = scheduler.conf["caliDataCR"][qIndex]["CRfreq"]

        # Add waves
        job = ham.createJob()

        crWave = quasiSquareErf(CRtg, CRamp, 0.1 * CRtg, 0.9 * CRtg, 0.35 * CRamp)
        job.appendWave(uWave, 1, crWave, freq=CRfreq)

        return job

    def generate1QGate(ham: QHamiltonian, cirLine, scheduler) -> QJob:
        """
        Generate the single qubit gate
        """
        gateName = cirLine.data.name
        job = ham.createJob()
        if gateName in ['X', 'RX', 'Y', 'RY']:

            qIndexL = list(cirLine.qRegIndexList)
            qIndex = qIndexL[0]
            if "caliDataXY" not in scheduler.conf.keys():
                raise ArgumentError(f"'caliDataXY' does not exist in scheduler.conf, "
                                    f"hence X/Y controls are not supported.")
            if qIndex not in scheduler.conf["caliDataXY"].keys():
                raise ArgumentError(f"Configuration of qubit(s) {qIndex} does not, "
                                    f"hence X/Y controls are not supported.")
            # Pulse parameters
            piLen = scheduler.conf["caliDataXY"][qIndex]["piLen"]
            piTau = scheduler.conf["caliDataXY"][qIndex]["piTau"]
            piSigma = scheduler.conf["caliDataXY"][qIndex]["piSigma"]

            if gateName in ['X', 'Y']:
                piAmp = scheduler.conf["caliDataXY"][qIndex]["piAmp"]
            elif gateName in ['RX', 'RY']:
                piAmp = cirLine.data.uGateArgumentList[0] / pi * scheduler.conf["caliDataXY"][qIndex]["piAmp"]
            else:
                raise ArgumentError(f"Unsupported gate {gateName}!")

            if gateName in ['X', 'RX']:
                phShift = 0
            elif gateName in ['Y', 'RY']:
                phShift = pi / 2
            else:
                raise ArgumentError(f"Unsupported gate {gateName}!")

            piWave = gaussian(piLen, piAmp, piTau, piSigma)
            job.appendWave(uWave, qIndex, piWave, phase0=phShift)

        elif gateName == 'VZ':

            job = generateVZ(ham, cirLine)

        else:
            raise ArgumentError(f"Unsupported gate {gateName}!")

        return job

    def generateEchoedCR(ham: QHamiltonian, cirLine, scheduler) -> QJob:
        """
        Generate echoed cross-resonance gate
        """
        qIndex = list(cirLine.qRegIndexList)
        jobCR = generateCR(ham=ham, cirLine=CircuitLine(CR, qIndex), scheduler=scheduler, tFactor=0.5)
        jobX = generate1QGate(ham=ham, cirLine=CircuitLine(X, [qIndex[0]]), scheduler=scheduler)
        jobCRMinus = generateCR(ham=ham, cirLine=CircuitLine(CR, qIndex), scheduler=scheduler,
                                ampFactor=-1, tFactor=0.5)

        return jobCR + jobX + jobCRMinus + jobX

    def generateCNOT(ham: QHamiltonian, cirLine, scheduler) -> QJob:
        """
        Generate the CNOT gate using echoed-CR gate and virtual-Z gate.
        """
        qIndex = list(cirLine.qRegIndexList)
        jobVZ = generateVZ(ham=ham, cirLine=CircuitLine(VZ(pi / 2), [qIndex[0]]))
        jobEchoedCR = generateEchoedCR(ham=ham, cirLine=CircuitLine(CR, qIndex), scheduler=scheduler)
        jobRX = generate1QGate(ham=ham, cirLine=CircuitLine(RX(pi / 2), [qIndex[1]]), scheduler=scheduler)

        return jobVZ + jobEchoedCR + jobRX

    generator.addGenerator(['CR'], generateCR)
    generator.addGenerator(['X', 'Y', 'RX', 'RY', 'VZ'], generate1QGate)
    generator.addGenerator(['EchoedCR'], generateEchoedCR)
    generator.addGenerator(['CNOT'], generateCNOT)
    generator.addGenerator(['H'], generateHadamard)

    return generator


def pulseSimQCQ(dt: float = 0.01, frameMode: str = 'lab') -> PulseModel:
    r"""
    Return a template of QCQ model.

    :param dt: a sampling time period.
    :param frameMode: indicates the frame, ``rot`` indicates the rotating frame,
        ``lab`` indicates the lab frame.

    :return: a QCQ PulseModel object.
    """

    # Parameters setting
    couplerLevel = 2
    qubitLevel = 2

    subSysNum = 3
    subSysLevel = [couplerLevel, qubitLevel, qubitLevel]  # Coupler - Control Qubit - Target Qubit

    freqDict = {
        0: 6.3 * (2 * pi),  # Coupler frequency (MHz)
        1: 5.1 * (2 * pi),  # Control qubit frequency (MHz)
        2: 4.9 * (2 * pi)  # Target qubit frequency (MHz)
    }

    anharmDict = {
        0: 0.0 * (2 * pi),  # Coupler anharmonicity (MHz)
        1: -0.33 * (2 * pi),  # Control qubit anharmonicity (MHz)
        2: -0.33 * (2 * pi)  # Target qubit anharmonicity (MHz)
    }

    couplingMap = {
        (0, 1): 0.098 * (2 * pi),  # Coupling strength of coupler and control qubit (MHz)
        (0, 2): 0.083 * (2 * pi),  # Coupling strength of coupler and target qubit (MHz)
        (1, 2): 0.0025 * (2 * pi)  # Coupling strength of control qubit and target qubit (MHz)
    }

    model = PulseModel(subSysNum=subSysNum, sysLevel=subSysLevel, qubitFreq=freqDict, qubitAnharm=anharmDict,
                       driveFreq=freqDict, couplingMap=couplingMap, frameMode=frameMode, pulseGenerator=pulseGenerator,
                       dt=dt)

    model.savePulse = False

    # Calibrated drive frequency for each qubit

    wd0 = 6.3 * (2 * pi)
    wd1 = 31.986542344407095
    wd2 = 30.768373764443712

    if frameMode == 'lab':
        # Single-qubit gate calibration data
        model.conf["caliDataXY"] = {
            1: {"piAmp": 0.2946938775510204, "piLen": 30, "piTau": 30 / 2, "piSigma": 30 / 7, "dragCoef": 0.0},
            2: {"piAmp": 0.29428571428571426, "piLen": 30, "piTau": 30 / 2, "piSigma": 30 / 7, "dragCoef": 0.0}
        }

        # Calibration data for echo CR gate
        model.conf["caliDataCR"] = {
            (1, 2): {"CRtg": 273, "CRamp": 0.15 * 2 * pi / 2, "CRfreq": 4.8948 * 2 * pi - wd1}
        }

        # Set the local oscillator

        model.ham.job.setLO(uWave, 0, freq=wd0)
        model.ham.job.setLO(uWave, 1, freq=wd1)
        model.ham.job.setLO(uWave, 2, freq=wd2)

    elif frameMode == 'rot':
        raise Error.ArgumentError("Only lab frame is supported!")

    else:
        raise Error.ArgumentError("Only lab frame is supported!")

    return model


def computationalSubspace(obj: Union[PulseModel, QHamiltonian]) -> ndarray:
    r"""
    Return the computational subspace matrix

    :param obj: input PulseModel object or QHamiltonian object

    :return: a 4-dimensions ndarray
    """
    if isinstance(obj, PulseModel):
        ham = obj.createQHamiltonian(frameMode='lab')
    elif isinstance(obj, QHamiltonian):
        ham = obj
    else:
        raise Error.ArgumentError("Pleases input a PulseModel object or a QHamiltonian object.")

    if ham.sysLevel == [3, 4, 4]:

        ham.buildCache()
        matQCQ = copy.deepcopy(ham.driftCache)

        # Decoupling coupler
        matQC1Q, unitaryC1 = fastBlockDiagonalization(mat=matQCQ, blockDim=2 * (4 ** 2))
        matQC0Q, unitaryC0 = fastBlockDiagonalization(mat=matQC1Q, blockDim=(4 ** 2))

        # Rearrange
        matQQ = rearrange(matQC0Q[:(4 ** 2), :(4 ** 2)])

        # Obtain the computation subspace matrix
        subMatrix = matQQ[:4, :4]

    else:
        raise Error.ArgumentError("Coupler (index: 0) level should be 3 and Qubit (index: 1, 2) level should be 4.")

    return subMatrix


def staticZZ(obj: Union[PulseModel, QHamiltonian]) -> float:
    r"""
    Compute the static ZZ strength of the given QCQ model

    :param obj: input a PulseModel object or QHamiltonian object

    :return: a float
    """

    if isinstance(obj, PulseModel):
        ham = obj.createQHamiltonian(frameMode='lab')
    elif isinstance(obj, QHamiltonian):
        ham = obj
    else:
        raise Error.ArgumentError("Pleases input a PulseModel object or a QHamiltonian object.")

    if ham.sysLevel == [3, 4, 4]:

        ham.buildCache()
        matQCQ = copy.deepcopy(ham.driftCache)

        # Create Qubit-Qubit subspace basis vectors
        vec00 = tensor(basis(3, 0), basis(4, 0), basis(4, 0))
        vec01 = tensor(basis(3, 0), basis(4, 0), basis(4, 1))
        vec10 = tensor(basis(3, 0), basis(4, 1), basis(4, 0))
        vec11 = tensor(basis(3, 0), basis(4, 1), basis(4, 1))

        # Get the eigen system of QCQ Hamiltonian
        originEigenVals, originEigenVecs = eigenSystem(matQCQ)

        # vector indexes of QQ eigenvectors
        idx00 = findIndex(originEigenVecs, [vec00])[0]
        idx10 = findIndex(originEigenVecs, [vec10])[0]
        idx01 = findIndex(originEigenVecs, [vec01])[0]
        idx11 = findIndex(originEigenVecs, [vec11])[0]

        staticZZCrossTalk = originEigenVals[idx00] - originEigenVals[idx10] - \
                            originEigenVals[idx01] + originEigenVals[idx11]

    else:
        raise Error.ArgumentError("Coupler (index: 0) level should be 3 and Qubit (index: 1, 2) level should be 4.")

    return staticZZCrossTalk


def effectiveCoupling(obj: Union[PulseModel, QHamiltonian]) -> float:
    r"""
    Return effective coupling strength of Qubit-Qubit Model

    :param obj: input PulseModel object or QHamiltonian object

    :return: a float
    """

    matrixXY = (tensor(sigmaX().matrix, sigmaX().matrix) + tensor(sigmaY().matrix, sigmaY().matrix)) / 2
    subMatrix = computationalSubspace(obj)

    effectiveJ = trace(matrixXY @ subMatrix).real / 2

    return effectiveJ


def dressedQubitFreq(obj: Union[PulseModel, QHamiltonian]) -> Dict:
    """
    Return the dressed frequencies of the control qubit and the target qubit

    :param obj: input a PulseModel object or QHamiltonian object

    :return: dressed frequencies dictionary
    """
    if isinstance(obj, PulseModel):
        ham = obj.createQHamiltonian(frameMode='lab')
    elif isinstance(obj, QHamiltonian):
        ham = obj
    else:
        raise Error.ArgumentError("Pleases input a PulseModel object or a QHamiltonian object.")

    if ham.sysLevel == [3, 4, 4]:

        ham.buildCache()
        matQCQ = copy.deepcopy(ham.driftCache)

        # Create Qubit-Qubit subspace basis vectors
        vec00 = tensor(basis(3, 0), basis(4, 0), basis(4, 0))
        vec01 = tensor(basis(3, 0), basis(4, 0), basis(4, 1))
        vec10 = tensor(basis(3, 0), basis(4, 1), basis(4, 0))
        vec11 = tensor(basis(3, 0), basis(4, 1), basis(4, 1))

        # Get the eigen system of QCQ Hamiltonian
        originEigenVals, originEigenVecs = eigenSystem(matQCQ)

        # Vector indexes of QQ eigenvectors
        idx00 = findIndex(originEigenVecs, [vec00])[0]
        idx10 = findIndex(originEigenVecs, [vec10])[0]
        idx01 = findIndex(originEigenVecs, [vec01])[0]
        idx11 = findIndex(originEigenVecs, [vec11])[0]

        dressedCtrlFreq = (originEigenVals[idx10] - originEigenVals[idx00]
                           + originEigenVals[idx11] - originEigenVals[idx01]) / 2

        dressedTargetFreq = (originEigenVals[idx11] - originEigenVals[idx10]
                             + originEigenVals[idx01] - originEigenVals[idx00]) / 2

        dressedFreqDict = {
            "control": dressedCtrlFreq,
            "target": dressedTargetFreq
        }

    else:
        raise Error.ArgumentError("Coupler (index: 0) level should be 3 and Qubit (index: 1, 2) level should be 4.")

    return dressedFreqDict


def pauliCoefficient(obj: Union[PulseModel, QHamiltonian], drivingAmp: Optional[float] = None) -> Dict:
    r"""
    Return Pauli coefficients of the Hamiltonian with CR drive

    :param obj: input PulseModel object or QHamiltonian object
    :param drivingAmp: amplitude of CR pulse

    :return: Pauli coefficients dictionary
    """

    def _diagnoalize(mat: ndarray):
        """
        Diagonalize the given matrix

        :param mat: input square matrix

        :return: diagonal matrix and the corresponding transformation unitary
        """
        matCache = copy.deepcopy(mat)
        eigenVals, eigenVecs = eigenSystem(matCache)
        indexVecList = createBasis(mat.shape[0], mat.shape[0])
        indexList = findIndex(eigenVecs, indexVecList)
        diagMat = diag(eigenVals[indexList])
        unitary = eigenVecs[:, indexList]
        return diagMat, unitary

    # Define pauli operator
    pauliOpZZ = tensor(sigmaZ().matrix, sigmaZ().matrix)
    pauliOpZX = tensor(sigmaZ().matrix, sigmaX().matrix)
    pauliOpZY = tensor(sigmaZ().matrix, sigmaY().matrix)
    pauliOpZI = tensor(sigmaZ().matrix, identity(2))
    pauliOpIZ = tensor(identity(2), sigmaZ().matrix)
    pauliOpIX = tensor(identity(2), sigmaX().matrix)
    pauliOpIY = tensor(identity(2), sigmaY().matrix)

    # Define the cross-resonance driving operator
    driveCRop = tensor((create(4).matrix + destroy(4).matrix),
                       identity(4))

    # Initial Hamiltonian
    if isinstance(obj, PulseModel):
        ham = obj.createQHamiltonian(frameMode='lab')
    elif isinstance(obj, QHamiltonian):
        ham = obj
    else:
        raise Error.ArgumentError("Pleases input a PulseModel object or a QHamiltonian object.")

    if ham.sysLevel == [3, 4, 4]:

        ham.buildCache()
        matQCQ = copy.deepcopy(ham.driftCache)

        # Decoupling coupler
        matQC1Q, unitaryC1 = fastBlockDiagonalization(mat=matQCQ, blockDim=2 * (4 ** 2))
        matQC0Q, unitaryC0 = fastBlockDiagonalization(mat=matQC1Q, blockDim=(4 ** 2))

        # Rearrange
        matQQ = rearrange(copy.deepcopy(matQC0Q[:(4 ** 2), :(4 ** 2)]))
        driveRearrange = rearrange(driveCRop)

        # Diagonalize the Q-Q subspace matrix
        matDiagQQ, unitaryDiagQQ = _diagnoalize(matQQ)

        if drivingAmp is not None:
            # Transform driving term
            driveMat = dagger(unitaryDiagQQ) @ driveRearrange @ unitaryDiagQQ

            # Initialize rotating term
            dressFreq = (matDiagQQ[1, 1] - matDiagQQ[0, 0] + matDiagQQ[3, 3] - matDiagQQ[2, 2]) / 2
            numberOperator = diag(array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6]))

            rotMat = matDiagQQ + drivingAmp * 0.5 * driveMat - dressFreq * numberOperator
            computeBlockMat, computeSubT = computeBlockDiagonalize(rotMat, 4)
            computeMat = computeBlockMat[:4, :4]

            if obj.qubitFreq[1] >= obj.qubitFreq[2]:
                finalMat, T = blockDiag(computeMat, [0, 1])
            else:
                finalMat, T = blockDiag(computeMat, [2, 3])

            pauliDict = {
                'ZI': trace(pauliOpZI @ finalMat).real / 2,
                'IZ': trace(pauliOpIZ @ finalMat).real / 2,
                'ZZ': trace(pauliOpZZ @ finalMat).real / 2,
                'ZX': trace(pauliOpZX @ finalMat).real / 2,
                'IX': trace(pauliOpIX @ finalMat).real / 2,
                'ZY': trace(pauliOpZY @ finalMat).real / 2,
                'IY': trace(pauliOpIY @ finalMat).real / 2
            }
        else:
            finalMat = matDiagQQ[:4, :4]

            pauliDict = {
                'ZI': trace(pauliOpZI @ finalMat).real / 2,
                'IZ': trace(pauliOpIZ @ finalMat).real / 2,
                'ZZ': trace(pauliOpZZ @ finalMat).real / 2,
                'ZX': trace(pauliOpZX @ finalMat).real / 2,
                'IX': trace(pauliOpIX @ finalMat).real / 2,
                'ZY': trace(pauliOpZY @ finalMat).real / 2,
                'IY': trace(pauliOpIY @ finalMat).real / 2
            }
    else:
        raise Error.ArgumentError("Coupler (index: 0) level should be 3 and Qubit (index: 1, 2) level should be 4.")

    return pauliDict


def createBasis(dim: int, count: int) -> List[ndarray]:
    """
    Create basis vectors list

    :param dim: dimension of the Hilbert space
    :param count: the number of the basis vectors

    :return: a ndarray List
    """
    vecList = []
    for num in range(count):
        vec = basis(dim, num)
        vecList.append(vec)
    return vecList


def rearrange(mat: ndarray) -> ndarray:
    """
    Rearrange the Q-Q Hamiltonian matrix by the excitation number

    :param mat: input Hamiltonian matrix

    :return: rearranged matrix
    """
    if mat.shape == (16, 16):
        matCache = copy.deepcopy(mat)

        matCache[:, [2, 4]] = matCache[:, [4, 2]]
        matCache[[2, 4], :] = matCache[[4, 2], :]

        matCache[:, [3, 5]] = matCache[:, [5, 3]]
        matCache[[3, 5], :] = matCache[[5, 3], :]

        matCache[:, [5, 8]] = matCache[:, [8, 5]]
        matCache[[5, 8], :] = matCache[[8, 5], :]

        matCache[:, [7, 12]] = matCache[:, [12, 7]]
        matCache[[7, 12], :] = matCache[[12, 7], :]

        matCache[:, [6, 8]] = matCache[:, [8, 6]]
        matCache[[6, 8], :] = matCache[[8, 6], :]

        matCache[:, [7, 9]] = matCache[:, [9, 7]]
        matCache[[7, 9], :] = matCache[[9, 7], :]

        matCache[:, [7, 8]] = matCache[:, [8, 7]]
        matCache[[7, 8], :] = matCache[[8, 7], :]

        matCache[:, [11, 13]] = matCache[:, [13, 11]]
        matCache[[11, 13], :] = matCache[[13, 11], :]

    else:
        raise Error.ArgumentError("The matrix dimension should be (16, 16)")

    return matCache


def fastBlockDiagonalization(mat: ndarray, blockDim: int) -> [ndarray, ndarray]:
    """
    Block diagonalize the given matrix

    :param mat: original matrix
    :param blockDim: dimension of the first block of the matrix

    :return: the block diagonalized matrix
    """
    matCache = copy.deepcopy(mat)
    eigenVals, eigenVecs = eigenSystem(matCache)
    subVecList = createBasis(mat.shape[0], blockDim)
    indexList = findIndex(eigenVecs, subVecList)
    matBD, transMat = blockDiag(matCache, indexList)

    return matBD, transMat


def computeBlockDiagonalize(mat: ndarray, blockDim: int = 4) -> [ndarray, ndarray]:
    """
    Block diagonalize the qubit-qubit subspace matrix and get the computational subspace block matrix

    :param mat: Qubit-Qubit subspace matrix
    :param blockDim: dimension of the computational subspace matrix (default: 4)

    :return: the block diagonalized matrix and transformation unitary
    """
    matCache = copy.deepcopy(mat)
    eigenVals, eigenVecs = eigenSystem(matCache)
    subVecList = createBasis(mat.shape[0], blockDim)
    indexList = findIndex(eigenVecs, subVecList)

    # Fix findIndex error
    if len(indexList) == 4:
        if indexList[0] == indexList[1]:
            if indexList[0] == mat.shape[0] - 1:
                indexList[0] = mat.shape[0] - 2
            elif indexList[0] == 0:
                indexList[1] = 1
            else:
                leftDiff = abs(eigenVals[indexList[0] - 1] - eigenVals[indexList[0]])
                righrDiff = abs(eigenVals[indexList[0]] - eigenVals[indexList[0] + 1])
                if leftDiff < righrDiff:
                    indexList[0] = indexList[0] - 1
                else:
                    indexList[0] = indexList[0] + 1
        elif indexList[2] == indexList[3]:
            if indexList[2] == mat.shape[0] - 1:
                indexList[2] = mat.shape[0] - 2
            elif indexList[2] == 0:
                indexList[3] = 1
            else:
                leftDiff = abs(eigenVals[indexList[2] - 1] - eigenVals[indexList[2]])
                rightDiff = abs(eigenVals[indexList[2]] - eigenVals[indexList[2] + 1])
                if leftDiff < rightDiff:
                    indexList[2] = indexList[2] - 1
                else:
                    indexList[2] = indexList[2] + 1
    elif len(indexList) == 2:
        if indexList[0] == indexList[1]:
            leftDiff = abs(eigenVals[indexList[0] - 1] - eigenVals[indexList[0]])
            rightDiff = abs(eigenVals[indexList[0]] - eigenVals[indexList[0] + 1])
            if indexList[0] == indexList[1]:
                if indexList[0] == mat.shape[0] - 1:
                    indexList[0] = mat.shape[0] - 2
                elif indexList[0] == 0:
                    indexList[1] = 1
            else:
                if leftDiff < rightDiff:
                    indexList[0] = indexList[0] - 1
                else:
                    indexList[0] = indexList[0] + 1

    matBD, transMat = blockDiag(matCache, indexList)

    return matBD, transMat

