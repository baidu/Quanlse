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
Numba Accelerator
"""

from numba import jit
from numpy import ndarray, dot, diag, linalg, exp


@jit(cache=True, nopython=True, parallel=True)
def expm(mat: ndarray) -> ndarray:
    """
    Calculate the matrix exponential :math:`e^{A}` of the input matrix :math:`A`.

    :param mat: the input matrix.
    :return: the matrix exponential.
    """
    # Solving the eigen
    evals, evecs = linalg.eig(mat)
    expEvals = exp(evals)
    d2 = diag(expEvals)
    #  Pardon this godawful circumlocution.
    b = dot(evecs, d2)
    bt = b.transpose()
    at = evecs.transpose()
    et, residuals, rank, s = linalg.lstsq(at, bt)
    e = et.transpose()
    return e
