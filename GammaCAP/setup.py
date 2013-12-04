#!/usr/bin/env python

from distutils.core import setup

setup(name='gammacap',
      version='0.9.06',
      description='Gamma Ray Cluster Analysis Package',
      long_description='The gammacap package provides statistical tools and clustering algorithms for analysis of gamma-ray data and is an acronym for "Gamma-ray Clustering Analysis Package. It is actively maintained and written in pure python, requiring only standard scientific libraries.',
      author='Eric Carlson',
      author_email='erccarls@ucsc.edu',
      url='http://planck.ucsc.edu/gammacap',
      keywords=['Requires: numpy','Requires: scikit-learn','Requires: pyfits'], 
      #package_dir={'': './'},
      packages = ["GammaCAP","GammaCAP.BGTools","GammaCAP.Stats"]
     )