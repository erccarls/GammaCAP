GammaCAP: A Gamma-Ray Clustering Analysis Package
Introduction
Welcome to the homepage of GammaCAP, the Gamma-ray Clustering Analysis package.  This package is written in pure python and provides a  simple yet flexible python interface and even simpler command-line tools to compute clustering information for gamma-ray event data.  It is  oriented toward Fermi-LAT data set, but is also easily configured for any general data set requiring only space,  time, and spectral coordinates.

At the most basic level, GammaCAP employs the clustering algorithm DBSCAN (Density Based Spatial Clustering for Applications with Noise) to detect point sources in 2 (spatial) or 2+1 (time) dimensions, given a list of event data (e.g. lat/long/time/energy).  A variety of statistics are then computed such as significance over background (which can be used as a test-statistic), centroids, and bounding ellipses.   The pixel-less nature of DBSCAN provides very high sensitivity to sparse gamma-ray data which can hopefully be used to gain a better understanding of the unresolved Isotropic Gamma-Ray Background (IGRB), improve the depth of point source catalogs, and provide an additional tool for distinguishing point source from diffuse emission classes (see e.g. arXiv:1304.5524 and arXiv:1210.0522 for sample applications of the algorithm).  In addition, the incorporation of timing information presents a novel method to search for transient sources that may otherwise be washed out by an integrated background signal.

Some of the other features included in GammaCAP:

-Use of Fermi instrument response functions and Diffuse Galactic and Isotropic Emission Models to automatically and adaptively tune DBSCAN parameters.
-Post-processing of cluster data using Boosted Decision Tree classification to lower background contamination and improve sensitivity ***(Still in progress).
-Source Ellipse Calculations using principle component analysis.
-Highly optimized cluster computation routines.
-Automatic parsing of Fermi-LAT data - just supply the events in the standard FITS format or directly supply coordinates.
-Simple simulation tools to quickly test sensitivity to new source models (also supports the Fermi Science-Tools gtobssim).
-Full PDF and online searchable HTML documentation of API.
-Tutorials highlighting the primary functionality.
Useful new feature requests are always appreciated!

Installation
Through python setup tools (Linux/Win/OSX)
Although GammaCAP is pure python and should support Windows and Mac provided the dependencies are met, it was developed and tested on Linux (Ubuntu 12.04). The easiest way to install GammaCAP is through python setup tools which will automatically install all required dependencies.  If you don't have python setup tools installed, you can get it here: https://pypi.python.org/pypi/setuptools
easy_install gammacap
**Note that superuser privileges will be required to install on the system python.

Upgrading to the latest version is just as simple...

easy_install --upgrade gammacap

Additional Files
In order to use some of the plotting functionality, the Basemap Matplotlib-Toolkit is required.  Instructions for installation can be found here.  On most linux distributions, this is likely included in the repository and should be installed there.  On Ubuntu, the relevant command is,
sudo apt-get install python-mpltoolkits.basemap

Finally, the Fermi diffuse galactic background model must be downloaded if the Fermi Science Tools are not installed.  If the Fermi Science Tools are installed, and $FERMI_DIR is a valid environmental variable pointing to the installation directory, then this file should be found automatically.  Otherwise it can be downloaded on the Fermi website (below) and you will need to provide the path to GammaCAP as mentioned in the tutorials.

Diffuse Galactic Model (~500 MB)
http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/gll_iem_v05.fit

Installation From Source
If an alternative installation is needed, the latest source tarballs and windows executables are also available here

Tarballs and Executables

This package has the following dependencies.

numpy
pyfits
scikit-learn
matplotlib
Bug Reports and Feature Requests
GammaCAP is being actively developed and feature requests are encouraged provided they will be useful to others.  Please submit these and any bugs at the Github issues page here: https://github.com/erccarls/GammaCAP/issues



Tutorials & Documentation
Documentation

GammaCAP: A Gamma-ray Clustering Analysis Package -- A  description/study of the DBSCAN algorithm in application to Fermi data, as well as details of the cluster properties.

Software Reference Manual ( online | pdf )

Tutorials

SimTools Tutorial ( online | .ipynb | .py | .pdf ) -- Learn how to simulate Fermi-LAT data and easily implement point source models.

DBSCAN 2D Tutorial ( online | .ipynb | .py | .pdf ) -- Learn how to use DBSCAN to compute spatial clustering information on Fermi-LAT data and simulations.

DBSCAN 3D Tutorial ( online | .ipynb | .py | .pdf ) -- Learn how to use DBSCAN to compute spatio-temporal clustering information on Fermi-LAT data and simulations.


Contact
Questions should be directed to Eric Carlson. Bug reports are preferentially submitted through Github here if possible, but will also be answered via email.

email: erccarls@ucsc.edu
