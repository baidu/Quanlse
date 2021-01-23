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
Plot functions.
"""

import numpy, math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogFormatter
from typing import List, Dict, Any, Union
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Quanlse.Utils.Tools import computationalBasisList


def plotScheduler(taskResult: Dict[str, Any], dark: bool = False) -> None:
    """
    Plot the pulse scheduling figures using the taskResult from the QuanlseScheduler.

    :param taskResult: task result dictionary from QuanlseScheduler
    :param dark: enables a dark-themed mode
    """

    # get qubit number
    qubits = numpy.shape(taskResult["plot"]["tarray_plot"])[0]

    # convert amplitudes and times to 2-dim lists
    tArrayPlot = taskResult["plot"]["tarray_plot"]
    fArrayPlot = taskResult["plot"]["farray_plot"]
    fZArrayPlot = taskResult["plot"]["fzarray_plot"]

    # initialize lists that will be used as function parameters
    x = []
    y = []
    colors = []
    title = []
    yLabel = []

    # set pulse colors
    if dark:
        xyColor = 'pink'
        zColor = 'lightblue'
    else:
        xyColor = 'red'
        zColor = 'green'

    for i in numpy.arange(qubits):
        # pass in values for title/label/color lists
        title.append('Qubit %d' % i + ' - XY Channel')
        title.append('Qubit %d' % i + ' - Z Channel')
        yLabel.append('Amp (a.u.)')
        yLabel.append('Amp (a.u.)')
        colors.append(xyColor)
        colors.append(zColor)

        # pass in current pulses into lists
        x.append(list(tArrayPlot[i]))
        x.append(list(tArrayPlot[i]))
        y.append(list(fArrayPlot[i]))
        y.append(list(fZArrayPlot[i]))

    # to make the lists of the same size/range
    newX = x
    newY = y

    # check whether the input data is empty
    empty = True

    # find time range
    for i in range(len(x)):
        for j in range(len(x[i])):
            if empty:
                minX = x[i][j]
                maxX = x[i][j]
                empty = False
            if x[i][j] > maxX:
                maxX = x[i][j]
            if x[i][j] < minX:
                minX = x[i][j]
    assert not empty, "Please input non-empty array"

    # fill the pulses' blank space with zeros
    for i in range(len(x)):
        if len(x[i]) == 0:
            x[i] = [minX, maxX]
            y[i] = [0, 0]
        dt = x[i][1] - x[i][0]
        for j in range(int((min(x[i]) - minX) / dt)):
            newY[i].insert(0, 0)
        newX[i] = list(numpy.linspace(minX, min(x[i]), int((min(x[i]) - minX) / dt))) + newX[i]
        for j in range(int((maxX - max(x[i])) / dt)):
            newY[i].append(0)
        newX[i] = newX[i] + (list(numpy.linspace(max(x[i]), maxX, int((maxX - max(x[i])) / dt))))

    # plot the pulse
    plotPulse(newX, newY, xLabel='Time (ns)', yLabel=yLabel, title=title, color=colors, dark=dark)


def plotBarGraph(x: List[Any], y: List[float], title: str = "", xLabel: str = "", yLabel: str = "",
                 color: str = 'blue', lineWidth: float = 1.0, fontSize: float = 10, spacing: float = 0.1) \
        -> None:
    """
    Plot bar graphs with given X labels and Y values lists, this function also supports other optional
    parameters as shown below.

    :param x: a list of X labels for the graphs
    :param y: a list of Y coordinates for the bar graphs
    :param title: overall title of the graph
    :param xLabel: label on X axis
    :param yLabel: label on Y axis
    :param color: a color for all bars, default at blue (from mint, blue, red, green, yellow, black, pink, cyan,
        purple, darkred, orange, brown, pink and teal)
    :param lineWidth: line width index
    :param fontSize: font size index
    :param spacing: vertical/horizontal spacing index
    """
    # define color map
    cMap = colorMap()

    xList = numpy.arange(len(y))

    # Create figure instance
    plt.figure(1, (15, 10))

    # Set graph size and background color
    plt.rcParams["figure.figsize"] = (15, 10)
    plt.rcParams['axes.linewidth'] = lineWidth * 2

    # title and size
    if title is not None:
        plt.title(title, size=int(fontSize * 1.4), weight='bold')

    # label and size
    if xLabel != "":
        plt.xlabel(xLabel, size=int(fontSize * 1.2), weight='bold')
    if yLabel != "":
        plt.ylabel(yLabel, size=int(fontSize * 1.2), weight='bold')

    # Ticks size
    plt.xticks(fontsize=fontSize, weight='bold')
    plt.yticks(fontsize=fontSize, weight='bold')

    # adjust vertical and horizontal spacing
    plt.xlim([min(xList) - (max(xList) - min(xList)) * spacing, max(xList) + (max(xList) - min(xList)) * spacing])
    if min(y) >= 0:
        plt.ylim([min(y), max(y) + (max(y) - min(y)) * spacing])
    else:
        plt.ylim([min(y) - (max(y) - min(y)) * spacing, max(y) + (max(y) - min(y)) * spacing])

    # configure color and plot
    if color is None:
        plt.bar(xList, y, alpha=0.7, linewidth=lineWidth)
    else:
        for aColor in cMap:
            if aColor[0] == color:
                plt.bar(xList, y, color=aColor[1], alpha=1, linewidth=lineWidth)
    plt.xticks(xList, x)

    # draw the line of y = 0
    plt.axhline(y=0, color='#ccc', linestyle=':', linewidth=lineWidth)
    plt.show()


def plotLineGraph(x: Union[List[List[float]], List[float]], y: Union[List[List[float]], List[float]], title: str = "",
                  xLabel: str = "", yLabel: str = "",
                  legends: List[str] = None, color: List[str] = None,
                  lineWidth: float = 2.0,
                  fontSize: float = 10, spacing: float = 0.05, log=False) \
        -> None:
    """
    Plot line graph, with the given 2-dimensional lists of X and Y values,
    this function also supports other optional parameters as shown below.

    :param x: a 2-dim list of X coordinates for the line graphs
    :param y: a 2-dim list of Y coordinates for the line graphs
    :param title: overall title of the graph
    :param xLabel: label on X axis
    :param yLabel: label on Y axis
    :param legends: a list of strings - a label each string
    :param color: a list of colors - a color for each line (unspecified lines will be of default color)
        (from mint, blue, red, green, yellow, black, pink, cyan, purple, darkred, orange, brown, pink and teal)
    :param lineWidth: line width index
    :param fontSize: font size index
    :param spacing: vertical spacing index
    :param log: log, default at false

    """

    # Default color theme
    if color is None:
        color = ['blue', 'pink', 'cyan', 'mint', 'red']

    # take log
    if log:
        tmp = y
        y = numpy.log(numpy.array(y))
        if y.min() == -numpy.Infinity:
            y = tmp
            print("can't take log of zero, switch back to non-log mode")
    # Define Color Map
    cMap = colorMap()

    # Set graph size and background color
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams['axes.linewidth'] = lineWidth * 0.75

    # Check the type of the input of x and y
    if not isinstance(x[0], list):
        x = [x]
        y = [y]

    # Create figure instance
    plt.figure(1, (15, 10))

    # find max and min for spacing purposes
    maximumX = max(x[0])
    minimumX = min(x[0])
    maximumY = max(y[0])
    minimumY = min(y[0])

    # plot and look for global min/max
    for i in range(0, len(x)):
        if max(y[i]) > maximumY:
            maximumY = max(y[i])
        if min(y[i]) < minimumY:
            minimumY = min(y[i])
        if max(x[i]) > maximumX:
            maximumX = max(x[i])
        if min(x[i]) < minimumX:
            minimumX = min(x[i])

        if (color is None) or (i < 0) or (i >= len(color)):
            if legends is not None:
                plt.plot(x[i], y[i], label=legends[i], linewidth=lineWidth, alpha=1)
            else:
                plt.plot(x[i], y[i], linewidth=lineWidth)
        else:
            flag = False
            for aColor in cMap:
                if aColor[0] == color[i]:
                    flag = True
                    if legends is not None:
                        plt.plot(x[i], y[i], color=aColor[1], linewidth=lineWidth, label=legends[i], alpha=1)
                    else:
                        plt.plot(x[i], y[i], color=aColor[1], linewidth=lineWidth, alpha=1)
            if flag is False:
                raise ValueError('Color not found!')

    # spacing
    plt.xlim([minimumX - (maximumX - minimumX) * spacing, maximumX + (maximumX - minimumY) * spacing])
    plt.ylim([minimumY - (maximumY - minimumY) * spacing, maximumY + (maximumY - minimumY) * spacing])

    # Ticks size
    plt.xticks(fontsize=fontSize, weight='bold')
    plt.yticks(fontsize=fontSize, weight='bold')

    # title and size
    if title is not None:
        plt.title(title, size=int(fontSize * 1.9), weight='bold')

    # label and size
    if xLabel is not None:
        plt.xlabel(xLabel, size=int(fontSize * 1.4), weight='bold')
    if yLabel is not None:
        plt.ylabel(yLabel, size=int(fontSize * 1.4), weight='bold')

    plt.legend()
    plt.show()


def plotPulse(x: Union[List[List[float]], List[float]], y: Union[List[List[float]], List[float]],
              title: Union[List[str], str], yLabel: Union[List[str], str], color: Union[List[str], str],
              xLabel: str = '', lineWidth: float = 1.0, fontSize: float = 10, spacing: float = 0.1, dark: bool = False):
    """
    Plots pulse graphs, and is used in the plotWave() function in the Hamiltonian file. This function takes
    two mandatory two-dimensional lists of X and Y values. This function also supports other optional parameters
    as shown below.

    :param x: a 2-dim list consisting the time coordinates for each pulse
    :param y: a 2-dim list consisting the amplitudes for each pulse
    :param title: a list of titles, one for each pulse
    :param yLabel: a lists of y labels for each pulse
    :param xLabel: a string of x label (for all pulse)
    :param color: a list of colors for each pulse
        (from mint, blue, red, green, yellow, black, pink, cyan, purple, darkred, orange, brown, pink and teal)
        If pulse consists more than 250 cuts, switch to slice
    :param lineWidth: line width index
    :param fontSize: font size index
    :param spacing: vertical spacing index
    :param dark: enables a dark-themed mode
    """

    # Set color theme
    if dark:
        fontColor = 'ghostwhite'
        bgColor = '#1B1C21'
        axisColor = 'grey'
    else:
        fontColor = 'black'
        bgColor = 'white'
        axisColor = 'black'

    # Import colormap
    cMap = colorMap()

    # create fig and ax and subplots to plot
    if len(x) == 1:
        fig, ax1 = plt.subplots(len(x), 1, figsize=(16, len(x) * 1.5 + 3/4))
        ax = [ax1]
    else:
        fig, ax = plt.subplots(len(x), 1, figsize=(16, len(x) * 1.5 + 3/4))
    fig.patch.set_facecolor(bgColor)

    # loop through the list of graphs
    for k in range(0, len(x)):

        # if lengths of the two lists don't match
        assert len(x[k]) == len(y[k]), "Lengths of lists do not match"

        # f list x contains more than 250 slices, switch to line plot
        maxPieces = 250
        plotType = 'slice'
        if len(y[k]) > maxPieces:
            plotType = 'line'

        # check whether color exists in colormap
        colorFound = False

        # find according color in colormap
        for aColor in cMap:

            # if color found
            if aColor[0] == color[k]:
                colorFound = True

                # plot graph (line/bar)
                if plotType == 'line':
                    ax[k].axhline(linewidth=lineWidth * 1.3, color=axisColor, alpha=0.4, ls=':')
                    ax[k].plot(x[k], y[k], color=aColor[1], linewidth=lineWidth * 1.3)
                    ax[k].fill_between(x[k], 0, y[k], facecolor=aColor[1], alpha=0.05)
                elif plotType != 'line':
                    # shift pulse so that first pulse starts on 0
                    for i in x[k]:
                        i += (x[k][1] - x[k][0]) / 2
                    ax[k].bar(x[k], y[k], color=bgColor, alpha=1,
                              width=x[k][1] - x[k][0], edgecolor=aColor[1], linewidth=lineWidth * 1.3)
                    ax[k].bar(x[k], y[k], color=aColor[1], alpha=0.05,
                              width=x[k][1] - x[k][0], linewidth=0)
                break

        # if invalid color
        if not colorFound:
            raise ValueError('Color not found!')

        # adjust spacing between borders
        if max(y[k]) == min(y[k]):
            ax[k].set_ylim(min(y[k]) - (spacing * 10), max(y[k]) + (spacing * 10))
        else:
            ax[k].set_ylim(
                [min(y[k]) - (max(y[k]) - min(y[k])) * spacing * 2.5,
                 max(y[k]) + (max(y[k]) - min(y[k])) * (spacing + 0.3)])
        ax[k].set_xlim(min(x[k]), max(x[k]))

        # if last, add label, otherwise remove ticks,
        if k != len(x) - 1:
            ax[k].set_xticks([])
        else:
            ax[k].set_xlabel(xLabel, size=fontSize * 1.6)

        # set title for each graph
        ax[k].set_title(title[k] + "  ", fontdict={'fontsize': 15, 'fontweight': 'bold'},
                        y=0.75, loc='right', color=fontColor)

        # set y label
        ax[k].set_ylabel(yLabel[k], size=fontSize * 1.4)

        # set yTick decimal place
        ax[k].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        # set tick size
        ax[k].tick_params(labelsize=fontSize * 1.3)

        # set spines linewidth
        # if k == 0:
        ax[k].spines["top"].set_linewidth(1.5)
        ax[k].spines["bottom"].set_linewidth(1.5)
        ax[k].spines["left"].set_linewidth(1.5)
        ax[k].spines["right"].set_linewidth(1.5)

        # set spines color
        ax[k].spines["bottom"].set_color(axisColor)
        ax[k].spines["right"].set_color(axisColor)
        ax[k].spines["left"].set_color(axisColor)
        ax[k].spines["top"].set_color(axisColor)

        # set label and ticks color
        ax[k].xaxis.label.set_color(fontColor)
        ax[k].yaxis.label.set_color(fontColor)
        ax[k].tick_params(axis='x', colors=fontColor)
        ax[k].tick_params(axis='y', colors=fontColor)

        # set facecolor
        ax[k].set_facecolor(bgColor)

    # align y labels
    fig.align_ylabels(ax[:])
    plt.show()


def plotProcess(chi: Dict[str, Any]) -> None:
    """
    Generate the Process Tomography Plot in 3D. This function takes a mandatory dictionary Chi,
    which is directly generated from processTomography.

    :param chi: Chi Dictionary that is obtained from processTomography
    """

    # setup the figure and axes
    fig = plt.figure(figsize=(12, 6))
    ax2 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')

    numX = len(chi['XLabel'])
    numY = len(chi['YLabel'])

    # fake data
    _x = numpy.arange(numX)
    _y = numpy.arange(numY)
    _xx, _yy = numpy.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()

    top = numpy.reshape(chi['Imag'], numX * numY, )
    bottom = numpy.zeros_like(top)
    width = depth = 0.7

    # plot graph
    ax1.bar3d(x, y, bottom, width, depth, top, shade=True, color='r')
    ax1.set_xticks(numpy.linspace(0.5, numX - 0.5, numX))
    ax1.set_yticks(numpy.linspace(0.5, numY - 0.5, numY))
    ax1.w_xaxis.set_ticklabels(chi['XLabel'], fontsize=12)
    ax1.w_yaxis.set_ticklabels(chi['YLabel'], fontsize=12)
    ax1.set_title('Chi Matrix - Imaginary Parts', fontsize=18)
    ax1.set_zlim(-0.5, 0.5)
    top = numpy.reshape(chi['Real'], numX * numY, )
    bottom = numpy.zeros_like(top)

    ax2.bar3d(x, y, bottom, width, depth, top, shade=True, color='c')
    ax2.set_xticks(numpy.linspace(0.5, numX - 0.5, numX))
    ax2.set_yticks(numpy.linspace(0.5, numY - 0.5, numY))
    ax2.w_xaxis.set_ticklabels(chi['XLabel'], fontsize=12)
    ax2.w_yaxis.set_ticklabels(chi['YLabel'], fontsize=12)
    ax2.set_title('Chi Matrix - Real Parts', fontsize=18)
    ax2.set_zlim(-0.5, 0.5)
    plt.show()


def plotHeatMap(matrix: numpy.ndarray, xTicks: List = None, yTicks: List = None, xLabel: str = "", yLabel: str = "",
                useLog: bool = False, cMap: str = "cividis") -> None:
    """
    Plot a 2D heat map.

    :param matrix: the values of each square
    :param xTicks: ticks on X axis
    :param yTicks: ticks on Y axis
    :param xLabel: label on X axis
    :param yLabel: label on Y axis
    :param useLog: format ticks on a log scale
    :param cMap: indicate color_map
    """

    ax = plt.gca()

    if useLog:
        _matrix = numpy.log10(matrix)
        im = ax.imshow(_matrix, cmap=cMap, vmin=-10, vmax=0)
    else:
        _matrix = matrix
        im = ax.imshow(_matrix, cmap=cMap)

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(numpy.arange(numpy.shape(_matrix)[1]))
    ax.set_yticks(numpy.arange(numpy.shape(_matrix)[0]))
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if xTicks is None:
        ax.set_xticklabels(numpy.arange(numpy.shape(_matrix)[1]))
    else:
        ax.set_xticklabels(xTicks)

    if yTicks is None:
        ax.set_yticklabels(numpy.arange(numpy.shape(_matrix)[0]))
    else:
        ax.set_yticklabels(yTicks)

    im_ratio = numpy.shape(_matrix)[0] / numpy.shape(_matrix)[1]
    if useLog:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046 * im_ratio, pad=0.04)
        cbar.ax.set_ylabel(r"$\log_{10}(Population)$", rotation=-90, va="bottom")
    else:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046 * im_ratio, pad=0.04)
        cbar.ax.set_ylabel("Population", rotation=-90, va="bottom")

    valFormat = "{x:.2f}"
    threshold = im.norm(numpy.max(_matrix)) / 2.
    textColors = ("black", "white")
    kw = dict(horizontalalignment="center",
              verticalalignment="center")

    for i in range(numpy.shape(_matrix)[0]):
        for j in range(numpy.shape(_matrix)[1]):
            value = float(_matrix[i, j])
            kw.update(color=textColors[int(im.norm(value) < threshold)])
            im.axes.text(j, i, str(round(value, 2)), **kw)

    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.show()


def colorMap():
    """
    Default color setting.

    :return: Color map
    """
    return [
        ['mint', 'mediumspringgreen'],
        ['lightblue', 'deepskyblue'],
        ['blue', 'cornflowerblue'],
        ['yellow', 'yellow'],
        ['green', 'forestgreen'],
        ['red', 'crimson'],
        ['gold', 'gold'],
        ['black', 'black'],
        ['cyan', 'darkturquoise'],
        ['purple', 'darkviolet'],
        ['teal', 'teal'],
        ['darkred', 'darkred'],
        ['orange', 'orangered'],
        ['brown', 'peru'],
        ['pink', 'deeppink']

    ]
