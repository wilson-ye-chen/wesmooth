from setuptools import setup

setup(
    name='wesmooth',
    version='0.0.4',
    description='Wasserstein Exponential Smoothing',
    url='https://github.com/wilson-ye-chen/wesmooth',
    author='Wilson Ye Chen',
    license='MIT',
    packages=['wesmooth'],
    install_requires=['numpy', 'scipy', 'matplotlib']
    )
