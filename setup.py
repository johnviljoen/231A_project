"""
generate a python env with version 3.10, and pip install -e . on this and u
should be good to go.

ensure the mujoco binary is also installed onto the system you are running onto in the proper way:
https://github.com/google-deepmind/mujoco
"""

from setuptools import setup, find_packages

setup(name='231A_project',
    version='0.0.1',
    url='https://github.com/johnviljoen/231A_project/',
    author='John Viljoen',
    author_email='johnviljoen2@gmail.com',
    install_requires=[
        'casadi',
        'neuromancer',
        'matplotlib',
        'mujoco',       # for the uncertain black-box simulation, not really necessary
        'tqdm',         # just for pretty loops in a couple places
        'imageio',      # just for DPC callback animations
    ],
    packages=find_packages(
        include=[
            '231A_project.*'
        ]
    ),
)
