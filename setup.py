from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'kinematic_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.csv') + glob('config/*.yaml')),
        (os.path.join('share', package_name, 'maps'),
            glob('maps/*.csv') + glob('maps/*.yaml') + glob('maps/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Joel Siby',
    maintainer_email='joel.siby@e-yantra.org',
    description='Kinematic MPC controller for F1TENTH (CasADi + IPOPT)',
    license='MIT',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'kinematic_mpc_node = kinematic_mpc.mpc_node:main',
        ],
    },
)
