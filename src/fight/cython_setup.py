# -*-coding:utf-8 -*-
"""
@File      :   cython_setup.py
@Time      :   2025/04/17 09:47:28
@Author    :   Urvan Christen
@Version   :   1.0
@Contact   :   urvan.christen@gmail.com
@Desc      :   Cython setup file for the cython modules
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("fight.pyx", annotate=True), include_dirs=[np.get_include()]
)
