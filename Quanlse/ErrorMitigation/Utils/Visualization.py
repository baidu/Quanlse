# !/usr/bin/python3
# -*- coding: utf8 -*-
# Copyright (c) 2021 Baidu, Inc. All Rights Reserved.
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
Functions for plotting in ErrorMitigation module.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import List

from Quanlse.QHamiltonian import QHamiltonian
from Quanlse.QOperation import FixedGate
from Quanlse.Utils.Functions import expect
from Quanlse.QPlatform.Error import ArgumentError
from Quanlse.Define import outputPath


def polarVectorToState(vec) -> np.ndarray:
    """
    From the polar vector representation to the state representation.

    :param vec: polar vector, array_like, (3,)
    :return: density matrix: ndarray, (2, 2)
    """
    ops = [FixedGate.X.getMatrix(), FixedGate.Y.getMatrix(), FixedGate.Z.getMatrix()]
    return (np.identity(2) + vec[0] * ops[0] + vec[1] * ops[1] + vec[2] * ops[2]) / 2


def stateToPolarVector(state) -> List:
    """
    From the state representation to the polar vector representation.

    :param state: state vector or density matrix
    :return: polar vector
    """
    ops = [FixedGate.X.getMatrix(), FixedGate.Y.getMatrix(), FixedGate.Z.getMatrix()]
    return [expect(ops[i], state) for i in range(3)]


def plotZNESequences(expectationsRescaled: List[List], expectationsExtrapolated: List[List], expectationsIdeal: List,
                     fileName: str = None):
    """
    Plot a figure that shows the extrapolated results and noise-rescaling results, with the X-axis
    representing the size of quantum gate sequences and the Y-axis representing the expectation
    values of the given mechanical quantity.

    :param expectationsRescaled: a series of noise-rescaling expectation values with shape [n, d + 1]
    :param expectationsExtrapolated: a series of extrapolated expectation values with shape [n, d]
    :param expectationsIdeal: a series of ideal expectation values with shape [n,]
    :param fileName: file name
    :return: None
    """
    # shape: [numSeq, d + 1] --> [d + 1, numSeq]
    expectationsRescaled = np.array(expectationsRescaled).transpose()
    # shape: [numSeq, d] --> [d, numSeq]
    expectationsExtrapolated = np.array(expectationsExtrapolated).transpose()
    numSeq = expectationsRescaled.shape[1]
    lengthEachSequence = list(range(1, numSeq + 1))

    # plot results
    plt.figure(figsize=(10, 6))
    for d, Es in enumerate(expectationsRescaled):
        plt.plot(lengthEachSequence, Es, 'o-', label='{}-order rescaled'.format(d + 1))
    for d, Es in enumerate(expectationsExtrapolated):
        plt.plot(lengthEachSequence, Es, 'o--', label='{}-order extrapolated'.format(d + 1))
    plt.plot(lengthEachSequence, expectationsIdeal, 'k+--', label='Ideal')
    plt.xlabel('Size of Clifford sequence')
    plt.ylabel('Expectation value')
    plt.title('Richardson Extrapolation Result of Different Order')
    plt.legend()
    plt.xticks(lengthEachSequence)
    if fileName is not None:
        localFile = os.path.join(outputPath, fileName)
        plt.savefig(localFile)
    plt.show()


def plotRescaleHamiltonianPulse(rescaleCoes: List[float], hamList: List[QHamiltonian], numChannel: int,
                                figsize: tuple = (12, 6), title=None):
    """
    Plot each pulse channel of a series of time-rescaling Hamiltonians. Suppose len(rescaleCoes)=r and numChannel=n,
    this function will show a figure including r*c subfigures which are
    arranged in "r" rows and "c" columns.

    :param rescaleCoes: rescaling coefficients
    :param hamList: Hamiltonian list whose length is equal to `rescaleCoes`
    :param numChannel: number of pulse channels
    :param figsize: size of the entire figure
    :param title: super title of the entire figure
    :return: fig, axes
    """
    fig = plt.figure(figsize=figsize)
    axes = []
    colorList = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if len(rescaleCoes) != len(hamList):
        raise ArgumentError("Length of 'rescaleCoes' and length of 'hamList' are not equal!")
    grids = gridspec.GridSpec(len(rescaleCoes), numChannel)
    # dtList: e.g. [1, 1.25, ...]
    dtList = [ham.dt for ham in hamList]
    # tList: e.g. [1*16, 1.25*16, ...]
    tList = [ham.dt * ham.job.endTimeDt for ham in hamList]

    tMax = max(tList)
    driveNames = ['Drive X Pulse', 'Drive Y Pulse']
    if numChannel == 3:
        driveNames.append('Drive Z Pulse')

    for c, dr in enumerate(driveNames):
        for r, coe in enumerate(rescaleCoes):
            dt = dtList[r]
            ham = hamList[r]

            # ax lies in (row, col): (r, c)
            ax = plt.subplot(grids[r, c])

            # key: the name of pulse channel
            key = list(ham.ctrlCache.keys())[c]
            endDt = ham.job.endTimeDt
            As = [ham.job.waveCache[key][i] for i in range(endDt)]
            tBar = []
            Abar = []
            # construct tBar (list type) and Abar (list type) to plot a bar figure
            for i in range(endDt + 1):
                tBar.extend([i * dt, i * dt])
                if i == 0:
                    Abar.extend([0, As[i]])
                elif i == endDt:
                    Abar.extend([As[i - 1], 0])
                else:
                    Abar.extend([As[i - 1], As[i]])

            plt.fill(tBar, Abar, color=colorList[r], alpha=0.6)
            plt.plot(tBar, Abar, color=colorList[r], label='coe: {:.2f}'.format(coe))
            plt.legend(loc='upper right')

            plt.xlim(0, tMax)

            if r == 0:
                plt.title(key)
            if r == len(rescaleCoes) - 1:
                plt.xlabel('Times')

            axes.append(ax)

    fig.tight_layout()
    plt.suptitle(title)
    return fig, axes
