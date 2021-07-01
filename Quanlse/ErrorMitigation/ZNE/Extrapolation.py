#!/usr/bin/python3
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
This script implements various extrapolation methods such as the Richardson extrapolation and linear extrapolation.
"""

from scipy import optimize
import numpy as np
from collections import Iterable

from Quanlse.QPlatform import Error


def extrapolate(rescalingCoes: Iterable,
                expectations: Iterable,
                type: str = 'richardson',
                order: int = None,
                a0: float = None,
                sysFunc: bool = True) -> float:
    r"""
    Return the zero-noise extrapolation result (when the rescaling coefficient is zero) using different
    extrapolation strategies.


    :math:`\left\{\lambda_j, E_j \right\}_{1}^m \rightarrow\lim_{\lambda \to 0} E(\lambda)`

    :param rescalingCoes: Iterable, shape (m,).
        a series of rescaling coefficients (:math:`\lambda`)
    :param expectations: Iterable, shape (m,).
        a series of expectations corresponding to 'rescalingCoes'
    :param type: str, default: 'linear'.
        strategy for extrapolating, options: ['linear', 'polynomial', 'richardson', 'poly-exponential', 'exponential']
        or its simplified notations: ['linear', 'poly', 'richard', 'poly-exp', 'exp']
    :param order: int, default: None.
        default is None means that when using `poly-exp`-type or `poly`-type strategy,
        result of a specific order should specify its value
    :param a0: float, default: None.
        default is None means that when using 'poly-exp'-type strategy and the first term is unknown
    :param sysFunc: bool, default: True.
        True implies using the default np.polyfit method, otherwise, use the implemented linearRegression method
    :return: value of expectation when the rescaling coefficient is zero

    References
    ----------
    .. [1] Giurgica-Tiron, T., et al. (2020). Digital zero noise extrapolation for quantum error mitigation: 306-316.
    .. [2] Temme, K., et al. (2017). "Error mitigation for short-depth quantum circuits."
            Physical Review Letters 119(18): 180509.
    .. [3] Wikipedia, "Polynomial regression",
           https://en.wikipedia.org/wiki/Polynomial_regression
    .. [4] Wikipedia, "Polynomial interpolation",
           https://en.wikipedia.org/wiki/Polynomial_interpolation
    """
    typeOptional = ['linear', 'polynomial', 'richardson', 'poly-exponential', 'exponential']
    typeOptionalSimple = ['linear', 'poly', 'richard', 'poly-exp', 'exp']
    if len(rescalingCoes) != len(expectations):
        raise Error.ArgumentError('The length of "rescalingCoes" and "expectations" should be equal!')
    if (type not in typeOptional) and (type not in typeOptionalSimple):
        raise Error.ArgumentError('Please specify the "type" argument from {}'.format(typeOptional))
    # variables definition
    lams = np.array(rescalingCoes)
    Es = np.array(expectations)

    # number of samples
    m = len(lams)
    # extrapolating procedure
    if type == 'linear':
        if m == 2:
            # simply two-point extrapolating
            slope = (Es[1] - Es[0]) / (lams[1] - lams[0])
            res = Es[0] + slope * (0 - lams[0])
        else:
            # linear fitting
            if sysFunc:
                res = np.polyfit(lams, Es, 1)[-1]
            else:
                res = linearRegression(lams, Es)[0]
    elif type == 'richardson' or type == 'richard':
        # (m-1)th-order Richardson Extrapolation
        # 1) solve linear equations
        A = np.array([lams ** i for i in range(m)])
        b = np.zeros_like(lams)
        b[0] = 1
        # 2) compute extrapolated result
        coeffs = np.linalg.solve(A, b)
        res = sum(coeffs * Es)
    elif type == 'polynomial' or type == 'poly':
        if order is not None:
            # return extrapolation result of the specific order
            if order >= m - 1:
                raise Error.ArgumentError(
                    'Please limit value of "order" lower than "m-1" (m is length of "expectations")')
            # polynomial basis function transformation
            if sysFunc:
                res = np.polyfit(lams, Es, order)[-1]
            else:
                res = polyRegression(lams, Es, order)[0]
        else:
            # return extrapolation result of polynomial
            # order: [2, 3, ..., m-2] (list including m-3 elements)
            orders = range(2, m - 1)
            if sysFunc:
                res = [np.polyfit(lams, Es, d)[-1] for d in orders]
            else:
                res = [polyRegression(lams, Es, d)[0] for d in orders]
    elif type == 'poly-exponential' or type == 'poly-exp':
        if a0 is not None:  # a0 is already known
            sign = 1 if np.alltrue(Es - a0 > 0) else -1
            EsLog = np.log(np.abs(Es - a0))
            if order is not None:
                if order >= m - 1:
                    raise Error.ArgumentError('Please limit value of "order" '
                                              'lower than "m-1" (m is length of "expectations")')
                # polynomial basis function transformation
                if sysFunc:
                    z0 = np.polyfit(lams, EsLog, order)[-1]
                else:
                    z0 = polyRegression(lams, EsLog, order)[0]
                res = a0 + sign * np.exp(z0)
            else:
                # return extrapolation result of poly-exponential
                # order: [2, 3, ..., m-2] (list including m-3 elements)
                orders = range(2, m - 1)
                res = []
                for d in orders:
                    if sysFunc:
                        z0 = np.polyfit(lams, EsLog, d)[-1]
                    else:
                        z0 = polyRegression(lams, EsLog, d)[0]
                    res.append(a0 + sign * np.exp(z0))
        else:
            raise Error.ArgumentError("The {} extrapolation method with a0 "
                                      "is unknown is under construction.".format(type))
    elif type == 'exponential' or type == 'exp':
        if a0 is not None:  # a0 is already known
            sign = 1 if np.alltrue(Es - a0 > 0) else -1
            EsLog = np.log(np.abs(Es - a0))
            if m == 2:
                # simply two-point extrapolating
                slope = (EsLog[1] - EsLog[0]) / (lams[1] - lams[0])
                z0 = slope * (0 - lams[0]) + EsLog[0]
                res = a0 + sign * np.exp(z0)
            else:
                if sysFunc:
                    z0 = np.polyfit(lams, EsLog, 1)[-1]
                else:
                    z0 = linearRegression(lams, EsLog)[0]
                res = a0 + sign * np.exp(z0)
        else:
            # numeric fitting using scipy.optimize
            paras, _ = optimize.curve_fit(expSingle, lams, Es)
            res = paras[0] + paras[1]

    return res


def expSingle(x, a, b, c):
    r"""
    Function prototype used for optimize.curve_fit, with the expression :math:`a + b e^{- c x}`.
    """
    return a + b * np.exp(- c * x)


def linearRegression(x, y):
    r"""
    Linear regression method. Fit the given data points to the following linear model:

    :math:`y = \beta_0 + \beta_1     x`

    :param x: a list of independent variables
    :param y: a list of dependent variables
    :return: An array of coefficients: :math:`\beta_0` (intercept) and :math:`\beta_1` (slope)
    """
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    m = len(x)
    xAvg = np.mean(x)
    yAvg = np.mean(y)
    Sxx = np.sum(x ** 2) - m * xAvg ** 2
    Sxy = np.sum(x * y) - m * xAvg * yAvg
    return np.array([yAvg - xAvg * Sxy / Sxx, Sxy / Sxx])


def polyRegression(x: list, y: list, d: int):
    r"""
    Polynomial regression method to the d-th order. Fit the given data points to the following linear model:

    :math:`y = \sum_{i=0}^d a_i x^i`

    :param x: a list of independent variables
    :param y: a list of dependent variables
    :param d: the polynomial regression order
    :return: an array of polynomial regression coefficients :math:`[a_0, a_1, \cdots, a_d]`
    """
    x = np.array(x).ravel()
    y = np.array(y).ravel()
    X = []  # shape: [m, d+1]
    for k in range(d + 1):
        X.append(np.expand_dims(x, axis=1))
    X = np.concatenate(X, axis=1)
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
