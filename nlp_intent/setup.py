from setuptools import setup

package_name = 'nlp_intent'

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
    description='Lightweight NLP intent extraction using Ollama for robotics',
    license='MIT',
    entry_points={
        'console_scripts': [
            'intent_node = nlp_intent.intent_node:main',
            'turtle_controller = nlp_intent.turtle_controller:main',
        ],
    },
)
