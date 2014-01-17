#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup,find_packages

setup(name='gammacap',
      version='0.9.26',
      description='Gamma Ray Cluster Analysis Package',
      long_description='The gammacap package provides statistical tools and clustering algorithms for analysis of gamma-ray data and is an acronym for "Gamma-ray Clustering Analysis Package. It is actively maintained and written in pure python, requiring only standard scientific libraries.',
      author='Eric Carlson',
      author_email='erccarls@ucsc.edu',
      url='http://planck.ucsc.edu/gammacap',
      install_requires = ['numpy','scikit-learn','pyfits','matplotlib'],
      package_data = {'GammaCAP': ['./*.fits','./*.txt']},
      packages = find_packages()
      #packages = ["GammaCAP","GammaCAP.BGTools","GammaCAP.Stats"]
     )
