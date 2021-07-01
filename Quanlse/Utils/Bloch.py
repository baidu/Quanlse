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
Bloch Sphere Visualization Module
"""

import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np
from typing import Union

from Quanlse.QOperation import Error


def plotBloch(coordinates: list = None, xyzData: list = None, color: str ='Purple', mode: str ='animate',
              interval: float = None, save: bool = False, size: Union[int, float] = 2, title: str = '',
              fName: str = None, fps: float = 20):
    """
    This function plots Bloch sphere - supports 'animate' 'vector' and 'scatter' mode.

    :param coordinates: The list of Cartesian coordinates of n data points to plot.
    :param xyzData: The list of [XS YS ZS] of n data points to plot.
    :param color: The color of the point.
    :param mode: support 'animate' 'vector' and 'scatter' modes.
    :param interval: Delay between frames in ms.
    :param save: Whether to save animation as a gif file.
    :param size: Size of points in scatter mode.
    :param title: The title of the figure.
    :param fName: The name of the file saved.
    :param fps: The fps of gif file to save.
    """

    if coordinates is not None:
        coordinates = np.array(coordinates)
        coordinates[:, [0, 1]] = -1 * coordinates[:, [1, 0]]
        data = np.array(coordinates).T

    elif xyzData is not None:
        data = np.array(xyzData)
        data[[0, 1], :] = -1 * data[[1, 0], :]
        coordinates = data.T

    else:
        raise Error.ArgumentError('xyzData and coordinate unfilled.')

    # create figure
    fig = plt.figure(figsize=[6, 6])
    ax = Axes3D(fig)

    # Set X range
    ax.set_xlim3d([-1, 1])
    ax.set_xlabel('X')

    # Set Y range
    ax.set_ylim3d([-1, 1])
    ax.set_ylabel('Y')

    # Set Z range
    ax.set_zlim3d([-1, 1])
    ax.set_zlabel('Z')

    pNumber = 100

    # Plot Bloch Sphere
    u = np.linspace(0, 2 * np.pi, pNumber)
    v = np.linspace(0, np.pi, pNumber)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1.3 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, rstride=4, cstride=4, color='ghostwhite', linewidth=0, alpha=0.2)
    ax.plot(np.sin(u), np.cos(u), 0, color='grey')
    ax.plot([0] * pNumber, np.sin(u), 1.3 * np.cos(u), color='grey')

    # Hide grid lines
    ax.grid(False)

    # Hide default axis
    plt.axis('off')

    # Plot Axis and Axis Label
    ax.quiver(-1.2, 0, 0, 1.2, 0, 0, length=2, normalize=False,
              arrow_length_ratio=0.03, lw=1.5, color='black')
    ax.quiver(0, 1.2, 0, 0, -1.2, 0, length=2, normalize=False,
              arrow_length_ratio=0.03, lw=1.5, color='black')
    ax.quiver(0, 0, -1.2 * 1.3, 0, 0, 1.2 * 1.3, length=2, normalize=False,
              arrow_length_ratio=0.03, lw=2.5, color='black')
    ax.text(0, 0, 1.3 * 1.3, s='Z', weight='heavy')
    ax.text(0, -1.4, 0, s='x', weight='heavy')
    ax.text(1.3, 0, 0, s='y', weight='heavy')

    if mode == 'animate':

        data[2] = data[2] * 1.3

        line = plt.plot(data[0], data[1], data[2], lw=3)[0]

        def updateTrajectory(num, data, line):
            line.set_data(data[0:2, :num])
            line.set_3d_properties(data[2, :num])
            return line

        def update(num, data, line):
            update.vec.remove()
            update.vec = ax.quiver(0, 0, 0, data[0][num], data[1][num], data[2][num],
                                   normalize=False, arrow_length_ratio=0.05, lw=2, color=color)
            line = updateTrajectory(num, data, line)
            return line

        numData = len(coordinates)

        if interval is None:
            interval = 3 / numData

        update.vec = ax.quiver(0, 0, 0, 0, 0, 0)

        line_ani = animation.FuncAnimation(fig=fig, func=update, frames=numData,
                                           fargs=(data, line), interval=interval)

        plt.show()
        if save:
            writerGif = animation.PillowWriter(fps=fps)
            now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            line_ani.save(now + title + '.gif', writer=writerGif)

    elif mode == 'vector':
        # Vector mode
        coordinates = np.array(data).T
        for dat in coordinates:
            ax.quiver(0, 0, 0, dat[0], dat[1], dat[2] * 1.3,
                      normalize=False, arrow_length_ratio=0.1, lw=2, color=color)
        plt.title(label=title)
        if save:
            if fName is None:
                fName = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + title + '.png'
            plt.savefig(fName)
        plt.show()

    elif mode == 'scatter':
        # Scatter mode
        ax.scatter(data[0], data[1], data[2] * 1.3, marker='o', color=color, s=size)

        plt.title(label=title)
        if save:
            if fName is None:
                fName = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + title + '.png'
            plt.savefig(fName)
        plt.show()


def rho2Coordinate(rho: np.ndarray) -> list:
    """
    Convert a density matrix to the list of Cartesian coordinates.

    :param rho: The density matrix.
    :return: The list of Cartesian coordinates for given density matrix.
    """

    x1 = 2 * rho[0][1].real
    x2 = 2 * rho[1][0].imag
    x3 = (rho[0][0] - rho[1][1]).real

    return [x1, x2, x3]
