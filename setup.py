from distutils.core import setup
import setuptools

## will this be okay? there are images in my readme.md...
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='RCWA',
      version='1.0',
      description='rigorous coupled wave analysis module',
      author='Nathan Zhao',
      author_email='nzz2102@stanford.edu',
      url="https://github.com/zhaonat/Rigorous_Coupled_Wave_Analysis",
      packages=setuptools.find_packages(), ## gets all package dependencies
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ]
     )
# packages=['distutils', 'distutils.command'],

