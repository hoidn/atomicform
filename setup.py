#from ez_setup import use_setuptools
#use_setuptools()
from setuptools import setup, find_packages
#from distutils.core import setup

setup(name='atomicform',
      version='1.0',
      packages = find_packages('.'),
      package_dir={'atomicform': 'atomicform'},
      package_data={
            'atomicform': ['data/*'],
         },
      zip_safe = False
      )

