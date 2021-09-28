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
 Setup Installation
"""

from __future__ import absolute_import
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

setup(
    name='Quanlse',
    version='2.1.0',
    description='A cloud-based platform for quantum control.',
    author='Baidu Quantum',
    author_email='quantum@baidu.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'requests',
        'bce-python-sdk-reborn'
    ],
    python_requires='>=3.7, <4',
    license='Apache 2.0'
)