from setuptools import setup

setup(
name='Dian_MCMC',
version='0.1.0',
author='Dian Zhang',
author_email=' ',
packages=['Dian_MCMC', 'Dian_MCMC.HMC', "Dian_MCMC.MHMC", "Dian_MCMC.ProposalDistributions", "Dian_MCMC.StandardGaussianLikeli"],
url=' ',
license='LICENSE.txt',
description='Some basic MCMC Algorithms',
long_description=open('README.md').read(),
install_requires=[
    "numpy"
]
)