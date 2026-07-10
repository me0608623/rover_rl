from setuptools import setup

package_name = 'orca_filter'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/bystander.launch.py']),
        ('share/' + package_name + '/config', ['config/orca_params.yaml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='aa',
    maintainer_email='gpt0608626@gmail.com',
    description='ORCA (RVO2) safety filter - car deploy (non-cooperative)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'orca_bystander = orca_filter.bystander_node:main',
            'orca_safety = orca_filter.orca_safety_node:main',
        ],
    },
)
