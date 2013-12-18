python setup.py sdist
python setup.py register
python setup.py sdist bdist_wininst --plat-name=win32 upload
