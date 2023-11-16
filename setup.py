"""
generate a python env with version 3.9, and pip install -e . on this and u
should be good to go.
"""

from setuptools import setup, find_packages

setup(name='231A_project',
    version='0.0.1',
    url='https://github.com/johnviljoen/231A_project/',
    author='John Viljoen',
    author_email='johnviljoen2@gmail.com',
    install_requires=[
        'casadi',
        'torch',
        'matplotlib',
        'tqdm',
    ],
    packages=find_packages(
        include=[
            '231A_project.*'
        ]
    ),
)
