from setuptools import setup
from glob import glob

package_name = 'kinematic_mpc'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={package_name: '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.csv')),
        ('share/' + package_name + '/rviz',   glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='speed',
    maintainer_email='siddharth@e-yantra.org',
    description='Receding-horizon Kinematic MPC controller for F1TENTH.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'mpc_node = kinematic_mpc.mpc_node:main',
        ],
    },
)
