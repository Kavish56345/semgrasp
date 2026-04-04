from setuptools import setup

package_name = 'vlm_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='adminc',
    maintainer_email='admin@semgrasp.dev',
    description='VLM perception pipeline: GroundingDINO + MobileSAM + depth fusion',
    license='MIT',
    entry_points={
        'console_scripts': [
            'perception_node = vlm_perception.perception_node:main',
        ],
    },
)
