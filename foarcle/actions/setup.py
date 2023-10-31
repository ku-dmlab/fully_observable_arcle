
from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext

import os
import numpy as np

ext_modules=[
    Extension(f'foarcle.actions.{each}',
             [os.path.join("foarcle/actions", f'{each}.pyx')],
             include_dirs=[np.get_include()])
              for each in ['color', 'critical', 'o2actions', 'object']]

setup(name='package',
      packages=find_packages(),
      include_dirs=["foarcle/actions"],
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
     )
# Cython files can be compiled by python setup.py build_ext --inplace
