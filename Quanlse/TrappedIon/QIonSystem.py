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
Generally, when modeling a trapped ion quantum system, we need to define the chip probability.
A trapped ion quantum computing usually including trapped ion chip and laser control system,
below we define two class of a trapped ion system:

1: QLaser: in Quantum control of trapped ion, we need define QLaser basic information like laser wave length,
laser incident angle between the ion chain, how many laser pulse segments we used at time :math:`\tau`.
And in consideration of laser power limit, we restrict the max Rabi frequency of two Raman laser.

2: QIonChip: the basic qubit units in trapped ion quantum computing is the ions, where ion is a single atom which
loss one electron. If we constraint some ions by using special electronic trapped potential, we get a ion qubit chain
or lattice in ionChip. For a 1-D ion chain used by now, the basic quantum information is
what ion species in trap potential, how many ions in it, and what kind of trapped potential we used. By using these
basic parameters, we can get the chip quantum parameters like phonon ion coupling strength initial lamb-dicke parameters
and phonon frequency information.
"""

from typing import Dict, Any, Optional
from numpy import linspace, linalg, pi, identity, zeros
from numpy import cos, exp, ndarray
import numpy as np
from scipy.optimize import fsolve
from Quanlse.QPlatform import Error


class QLaser(object):
    """
    Basic class of control QLaser object. In trapped ion quantum control, the qubit is defined by internal energy of ion
    . Usually we use two laser beams to control one ion qubit, and the |0> |1> states can been connected by a Raman
    process, the parameter in effective Hamiltonian or Unitary gate formula of trapped ion partially depend on the
    laser control pattern.

    :param waveLength: laser wave length, the basic description of laser connected with laser frequency omega
                       by c/omega, c is the velocity of light
    :param laserAngle: the incidence angle between two Raman laser
    :param segments: the control laser slice number
    :param detuning: the laser frequency detuning between internal energy level, which is usually around MHz
    :param maxRabi: the max control power of two Raman laser, Limited by laser instrument performance in quantum lab
    :param seq: the Rabi frequency sequence list
    :param tg: the total control time of the laser sequence
    :param title: a user-given title to the QLaser object
    """

    def __init__(self, waveLength: float, laserAngle: float, segments: int,
                 detuning: float, maxRabi: float, seq: Any = None, tg: float = None, title=''):

        self.waveLength = waveLength  # type: float
        self.laserAngle = laserAngle  # type: float
        self.segments = segments  # type: int
        self.seq = seq  # type: Any
        self.tg = tg  # type: float
        self._args = {}  # type: Optional[Dict[str, Any]]
        self.detuning = detuning  # type: float
        self.maxRabi = maxRabi  # type: float
        self._RabiFrequency = {}  # type: Optional[Dict[str, Any]]
        self.title = title  # type: str

    def __str__(self) -> str:
        """
        Printer function for the QLaser object.
        """
        returnStr = ""

        # Print basic information
        returnStr += f"(1) Basic Information of laser control " \
                     f"`{'Raman process' if self.title == '' else self.title}`:\n"
        returnStr += f"    - Control laser wave length: {self.waveLength}nm\n"
        returnStr += f"    - Control laser detuning: {self.detuning / 1e6}MHz\n"
        returnStr += f"    - Laser incident angle: {round(self.laserAngle, 4)}\n"
        returnStr += f"    - Laser max power (effective max Rabi frequency): {self.maxRabi / 1e6}MHz\n"

        returnStr += f"(2) Laser sequence information:\n"
        returnStr += f"    - Laser segments: {self.segments}\n"
        returnStr += f"    - Laser total control time: {self.tg * 1e6}us\n"

        return returnStr

    @property
    def waveLength(self) -> float:
        """
        Return the wave length of control laser
        """
        return self._waveLength

    @waveLength.setter
    def waveLength(self, value: float):
        """
        Setting the control laser length
        """
        if not isinstance(value, float):
            raise Error.ArgumentError("The laser wave length must be float")
        if value < 100 or value > 2000:
            raise Error.ArgumentError("Please choose the available laser length, "
                                      "which in trapped ion quantum computing lab must between 100~2000nm")
        self._waveLength = value

    @property
    def laserAngle(self) -> float:
        """
        Return the laser incidence angle
        """
        return self._laserAngle

    @laserAngle.setter
    def laserAngle(self, value: float):
        """
        Setting the control laser incidence angle
        """
        if not isinstance(value, float):
            raise Error.ArgumentError("The laser incidence angle must be float")
        if value < 0 or value > np.pi:
            raise Error.ArgumentError("The laser incidence angle is between 0 ~ pi")
        self._laserAngle = value

    @property
    def segments(self) -> int:
        """
        Return the laser segments number
        """
        return self._segments

    @segments.setter
    def segments(self, value: int):
        """
        Setting the laser segments number
        """
        if not isinstance(value, int):
            raise Error.ArgumentError("The laser segments number must be integer")
        if value < 1 or value > 50:
            raise Error.ArgumentError("The laser segments number is between 2 ~ 50")
        self._segments = value

    @property
    def detuning(self) -> float:
        """
        Return the laser detuning
        """
        return self._detuning

    @detuning.setter
    def detuning(self, value: float):
        """
        Setting the laser detuning
        """
        if not isinstance(value, float):
            raise Error.ArgumentError("The laser detuning must be float")
        if value < 0:
            raise Error.ArgumentError("The laser detuning must larger than 0")
        self._detuning = value

    @property
    def maxRabi(self) -> float:
        """
        Return the laser max power
        """
        return self._maxRabi

    @maxRabi.setter
    def maxRabi(self, value: float):
        """
        Setting the laser max power
        """
        if not isinstance(value, float):
            raise Error.ArgumentError("The max Rabi frequency must be float")
        if value > 1e9:
            raise Error.ArgumentError("The max Rabi frequency must less than 1000 MHz")
        self._maxRabi = value

    def symmetrySeq(self, x: ndarray, ionNumber: int, indexIon: list) -> ndarray:
        """
        :param x: input list of optimization variables
        :param ionNumber: the ion number in chain
        :param indexIon: the ion implemented by laser
        :return: output omega-symmetry and phase-antisymmetry pulse sequence

        attention: if segments number is odd, this function would return an omega-symmetry and phase-antisymmetry pulses
        except the middle pulse piece
        """
        # symmetry operation
        halfSegments = self.segments / 2
        x = x.reshape((2, 2, int(np.ceil(halfSegments))))
        if self.segments % 2 == 0:
            p = np.array([[x[j][i][::-1] for i in range(len(x[0]))] for j in range(len(x))]) * 1
        else:
            r = np.array(
                [[x[j][i][0:int(np.ceil(halfSegments)) - 1] for i in range(len(x[0]))] for j in range(len(x))])
            p = np.array([[r[j][i][::-1] for i in range(len(x[0]))] for j in range(len(x))]) * 1
        p[0][1], p[1][1] = -p[0][1], -p[1][1]
        symSeq = np.array([np.concatenate((x[i], p[i]), axis=1) for i in range(len(x))])
        pulseSeq = np.zeros([ionNumber, 2, symSeq.shape[2]])
        pulseSeq[indexIon[0]] = symSeq[0] * 1.0
        pulseSeq[indexIon[1]] = symSeq[1] * 1.0
        return pulseSeq

    @property
    def waveVector(self):
        """
        Return the  laser waveVector of two Rabi laser
        """
        return 2 * pi * 2 * cos(self.laserAngle / 2) / (self.waveLength * 1e-9)


class QIonChip(object):
    """
    Basic class of ion chip. In trapped ion quantum computing, we need to define what kind of ion chip used in the lab,
    what kind of ion we use, and how many ion we load into ion chip, what kind of environment in the lab.
    There are different types of ion chip construction, like 1-dimension ion chip, 2-dimension ion chip,
    modular ion chip，multi-ion species type ion chip et.al. Aright now, we focus on the 1-dimension ion chain chip.

    :param ionMass: the ion species we used in the chip
    :param ionNumber: the total ion number loaded into ion chip
    :param temperature: the ion chip temperature
    :param title: a user-given title to the QIonChip object
    """
    def __init__(self, ionMass: int, ionNumber: int, temperature=None, title=''):
        self.ionMass = ionMass  # type: int
        self.ionNumber = ionNumber  # type: int
        self.temperature = temperature  # type: float
        self.title = title  # type: str
        self._energyLevel = {}  # type: Optional[Dict[str, Any]]


class QChain1D(QIonChip):
    """
    A basic 1-dimension ion chip, where is constructed by using special electronic trapped potential.

    :param trapZ: the transverse trap potential of ion chip
    :param trapXY: the axial trap potential of ion chip
    """
    def __init__(self, ionMass, ionNumber, temperature=None,
                 trapZ=None, trapXY=None) -> None:
        super(QChain1D, self).__init__(ionMass, ionNumber, temperature)
        self.trapZ = trapZ  # type: float
        self.trapXY = trapXY  # type: float
        self._property = None
        self._LambDicke = None

    def __str__(self) -> str:
        """
        Printer function for the ionChip object.
        """
        returnStr = ""

        # Print basic information
        returnStr += f"(1) Basic Information of ion chip `{'1-dimension chain' if self.title == '' else self.title}`:\n"
        returnStr += f"    - System ion species: {'Common atom element Yb' if self.ionMass == 171 else 'Other atom'}\n"
        returnStr += f"    - Number of ion qubits: {self.ionNumber}\n"
        returnStr += f"    - Chip temperature: {'1e-6' if self.temperature is None else self.temperature}K\n"

        returnStr += f"(2) Trapped potential information:\n"
        returnStr += f"    - Trapped potential in transverse direction: {round(self.trapZ / 1e6, 4)}MHz\n"
        returnStr += f"    - Trapped potential in axial direction: {round(self.trapXY / 1e6, 4)}MHz\n"

        return returnStr

    @property
    def ionMass(self) -> int:
        """
        Return the ion species in ion chip
        """
        return self._ionMass

    @ionMass.setter
    def ionMass(self, value: int):
        """
        Setting the ion species in chip
        """
        if not isinstance(value, int):
            raise Error.ArgumentError("The ion species must be integral")
        if value < 1 or value > 293:
            raise Error.ArgumentError("Please choose the nature atom species, "
                                      "which mass value must between 1~293")
        self._ionMass = value

    @property
    def ionNumber(self):
        """
        Return the ion number in the chip
        """
        return self._ionNumber

    @ionNumber.setter
    def ionNumber(self, value: int):
        """
        Setting ion number in the chip
        """
        if not isinstance(value, int):
            raise Error.ArgumentError("The ion number must be a integral")
        if value < 1 or value > 100:
            raise Error.ArgumentError("Trap device right now can just hold 1~100 ions in 1D chain")
        self._ionNumber = value

    @property
    def trapZ(self):
        """
        Return transverse trap potential value
        """
        return self._trapZ

    @trapZ.setter
    def trapZ(self, value: float):
        """
        Setting the transverse trap potential value
        """
        if not isinstance(value, float):
            raise Error.ArgumentError("Trap potential should be float number")
        if value < 0.0:
            raise Error.ArgumentError("The trap potential must larger the 0.0")
        self._trapZ = value
        self._property = None
        self._LambDicke = None

    @property
    def trapXY(self):
        """
        Return axial trap potential value
        """
        return self._trapXY

    @trapXY.setter
    def trapXY(self, value: float):
        """
        Setting the axial trap potential value
        """
        if not isinstance(value, float):
            raise Error.ArgumentError("Trap potential should be float number")
        if value < 0.0:
            raise Error.ArgumentError("The trap potential must larger the 0.0")
        if value < self.trapZ:
            raise Error.ArgumentError("The transverse trap potential must less than axial trap potential")
        self._trapXY = value
        self._property = None
        self._LambDicke = None

    @property
    def temperature(self):
        """
        Return the ion chip temperature
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        """
        Setting ion chip temperature
        """
        if value is None:
            self._temperature = 0.0
        elif not isinstance(value, float):
            raise Error.ArgumentError("The ion chip temperature must be a float number")
        elif value < 0.0:
            raise Error.ArgumentError("Temperature must larger than 0.0k")
        elif value > 1 or value < 1e-6:
            raise Error.ArgumentError("Ion chip temperature is between 1e-6K~1.0K")
        else:
            self._temperature = value

    def _ionPotential(self, u):
        """
        :param u: balance position of each ion
        :return: potential matrix for ion trap
        """
        potential = []
        for mIndex in range(self.ionNumber):
            s1 = 0
            s2 = 0
            for nIndex in range(mIndex):
                s1 = s1 + 1 / ((u[mIndex] - u[nIndex]) ** 2)

            for jIndex in range(mIndex + 1, self.ionNumber):
                s2 = s2 + 1 / ((u[mIndex] - u[jIndex]) ** 2)

            t = u[mIndex] - s1 + s2
            potential.append(t)
        return potential

    @property
    def ionPosition(self):
        """
        Return the equilibrium position of ion chain
        """
        u0 = linspace(-self.ionNumber ** 0.5, self.ionNumber ** 0.5, self.ionNumber)
        return fsolve(self._ionPotential, u0)

    @property
    def _ionPhononMode(self):
        """
        Calculate the phonon mode in ion chain
        """
        coulombAxial = zeros([self.ionNumber, self.ionNumber])
        u = self.ionPosition

        # generate the perturbation function of phonon axial mode
        for nIndex in range(self.ionNumber):
            for mIndex in range(self.ionNumber):
                if nIndex == mIndex:
                    s = 0
                    for pIndex in range(self.ionNumber):
                        if pIndex == mIndex:
                            s1 = 0
                        else:
                            s1 = 1 / abs((u[mIndex] - u[pIndex]) ** 3)
                        s = s + s1
                    avail = 2 * s + 1
                else:
                    avail = -2 / abs((u[mIndex] - u[nIndex]) ** 3)
                coulombAxial[[nIndex], [mIndex]] = avail

        # get axial mode and its frequency
        eigenAxial, vecAxial = linalg.eig(coulombAxial)
        idx = eigenAxial.argsort()[::1]
        axialFre, axialVec = self.trapZ * (eigenAxial[idx]) ** 0.5, vecAxial[:, idx]

        # get transverse mode and its frequency
        coulombTransverse = ((self.trapXY / self.trapZ) ** 2 + 1 / 2) * identity(self.ionNumber) - 1 / 2 * coulombAxial

        eigenTransverse, vecTransverse = linalg.eig(coulombTransverse)
        idx = eigenTransverse.argsort()[::1]
        if (self.trapZ / self.trapXY) ** 2 < 2 / (max(eigenAxial) - 1):
            transverseFre, transverseVec = self.trapZ * (eigenTransverse[idx]) ** 0.5, vecTransverse[:, idx]
        else:
            raise Error.ArgumentError("The 1D ion chain transverse phonon mode is unstable, "
                                      "please reduce trapZ and ion number or increase trapxy")

        return [axialFre, axialVec.T, transverseFre, transverseVec.T, coulombAxial, coulombTransverse]

    @property
    def phononMode(self):
        """
        Translate ion chip phonon frequency mode to class property
        """
        if self._property is None:
            self._property = self._ionPhononMode
        return self._property

    @property
    def ionAxialFre(self):
        """
        Return axial frequency of 1D chain
        """
        return self.phononMode[0]

    @ionAxialFre.setter
    def ionAxialFre(self, value: ndarray):
        """
        Return axial frequency of 1D chain
        """
        if value is None:
            raise Error.ArgumentError("You didn't input any axial frequency")
        elif not isinstance(value, ndarray):
            raise Error.ArgumentError("The ion chip axial frequency must be a float number")
        else:
            self._property[0] = value

    @property
    def ionAxialVec(self):
        """
        Return axial vector of 1D chain
        """
        return self.phononMode[1]

    @property
    def ionTransverseFre(self):
        """
        Return transverse frequency of 1D chain
        """
        return self.phononMode[2]

    @ionTransverseFre.setter
    def ionTransverseFre(self, value: ndarray):
        """
        Return axial frequency of 1D chain
        """
        if value is None:
            raise Error.ArgumentError("You didn't input any transverse frequency")
        elif not isinstance(value, ndarray):
            raise Error.ArgumentError("The ion chip transverse frequency must be a float number")
        else:
            print(self._property[2])
            print(type(self._property[2]))
            self._property[2] = value

    @property
    def ionTransverseVec(self):
        """
        Return transverse vector of 1D chain
        """
        return self.phononMode[3]

    @property
    def ionAxialMatrix(self):
        """
        Return transverse vector of 1D chain
        """
        return self.phononMode[4]

    @property
    def ionTransverseMatrix(self):
        """
        Return transverse vector of 1D chain
        """
        return self.phononMode[5]

    @property
    def transversePhononPopulation(self):
        """
        Return the phonon population
        """
        nk = [exp(- self.ionTransverseFre[i] / (1.3e11 * self.temperature)) /
              (1 - exp(- self.ionTransverseFre[i] / (1.3e11 * self.temperature)))
              for i in range(len(self.ionTransverseFre))]
        return nk

    @property
    def axialPhononPopulation(self):
        """
        Return the phonon population
        """
        nk = [exp(- self.ionAxialFre[i] / (1.3e11 * self.temperature)) /
              (1 - exp(- self.ionAxialFre[i] / (1.3e11 * self.temperature)))
              for i in range(len(self.ionAxialFre))]
        return nk

    @property
    def _ionChainHamiltonian(self):
        """
        Return: the value of phonon mode frequency and basic Lamb-Dicke parameter
        """
        hbarMass = 6.3e-8
        lambDickeAxial = zeros([self.ionNumber, self.ionNumber])  # initialize Lamb-Dicke parameters
        lambDickeTran = zeros([self.ionNumber, self.ionNumber])
        for ionIndex in range(self.ionNumber):
            lambDickeAxial[[ionIndex], :] = self.ionAxialVec[[ionIndex], :] * (
                    (hbarMass / (2 * self.ionMass * self.ionAxialFre)) ** 0.5)
        for ionIndex in range(self.ionNumber):
            lambDickeTran[[ionIndex], :] = self.ionTransverseVec[[ionIndex], :] * (
                    (hbarMass / (2 * self.ionMass * self.ionTransverseFre)) ** 0.5)
        return [lambDickeAxial, lambDickeTran]

    @property
    def LambDicke(self):
        if self._LambDicke is None:
            self._LambDicke = self._ionChainHamiltonian
        return self._LambDicke

    @property
    def lambDickeAxial(self):
        return self.LambDicke[0]

    @property
    def lambDickeTran(self):
        return self.LambDicke[1]
