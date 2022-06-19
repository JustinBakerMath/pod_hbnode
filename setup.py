from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
name='pod_hbnode',
version='1.0',
description='Proper orthogonal decomposition for accelerated neural ode training.',
author='Justin Baker',
author_email='baker@math.utah.edu',
packages=['pod_hbnode'],  #same as name
install_requires=['gdown', 'pandas', 'numpy', 'matplotlib', 'torchdiffeq', 'torch'], #external packages as dependencies
data_files=[('./out/', [])]
)
