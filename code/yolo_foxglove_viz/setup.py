import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'yolo_foxglove_viz'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(
            os.path.join('launch', '*launch.[pxy][yma]*')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jose Luis MV',
    maintainer_email='jlmv.96@gmail.com',
    description='This packages converts yolo_msgs to be visualized in Foxglove',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "yolo_to_foxglove = yolo_foxglove_viz.yolo_viz:main",
        ],
    },
)
