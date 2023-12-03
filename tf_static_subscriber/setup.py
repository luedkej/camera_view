from setuptools import find_packages, setup

package_name = 'tf_static_subscriber'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aidara',
    maintainer_email='luedkej@ethz.ch',
    description='subscriber to the TransfromStamped message from /tf_static',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
         'listener = tf_static_subscriber.Subscriber_to_Topic_tf_static:main',
        ],
    },
)
