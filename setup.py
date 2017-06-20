#from ez_setup import use_setuptools
#use_setuptools()
from __future__ import absolute_import
from setuptools import setup, find_packages
#from distutils.core import setup

setup(name=u'atomicform',
      version=u'1.0',
      packages = find_packages(u'.'),
      package_dir={u'atomicform': u'atomicform'},
      package_data={
            u'atomicform': [u'data/*'],
         },
      zip_safe = False
      )

