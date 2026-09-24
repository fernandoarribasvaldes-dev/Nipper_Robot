from setuptools import find_packages, setup

package_name = 'nipper_robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','opcua', 'asyncua'],
    zip_safe=True,
    maintainer='sarthakshirke',
    maintainer_email='sarthakshirke@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "diff_drive_controller_test = nipper_robot_controller.diff_drive_controller:main",
            "wheel_test = nipper_robot_controller.wheel_controller:main",
            "diff_drive_sim_test = nipper_robot_controller.diff_drive_sim_test:main",
            "collect_data = nipper_robot_controller.collecting_data:main",
            "diff_drive_sim_test_graph = nipper_robot_controller.diff_drive_sim_test_graph:main",
            "collect_data_all_wheels = nipper_robot_controller.collecting_data_all_wheels:main",
            "test_simulation = nipper_robot_controller.simulation_test:main",
            "odom_plot = nipper_robot_controller.diff_drive_sim_odom_plot:main",
            "diff_drive_controller_test_cobots = nipper_robot_controller.diff_drive_controller_cobots:main",
            "nipper_sim_graph = nipper_robot_controller.nipper_sim_shadow_graph:main",
            "nipper_estimated_cmd_vel = nipper_robot_controller.nipper_estimated_cmd_vel:main",
            "compare_sim_real = nipper_robot_controller.compare_sim_nipper:main",
            "pallet_detection_ros2 = nipper_robot_controller.pallet_detection_ros2:main",
            "nipperv6_controller = nipper_robot_controller.nipperv6_controller:main",
        ],
    },
)
