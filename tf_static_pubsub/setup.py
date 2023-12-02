from setuptools import find_packages, setup

package_name = 'tf_static_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'geometry_msgs'],
    zip_safe=True,
    maintainer='jakob',
    maintainer_email='luedkej@ethz.ch',
    description='publisher/subscriber_transformStamped',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'talker = tf_static_pubsub.publisher_member_function:main',
        	'listener = tf_static_pubsub.subscriber_member_function:main',
        ],
    },
)
