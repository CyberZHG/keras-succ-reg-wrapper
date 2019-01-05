from setuptools import setup, find_packages

setup(
    name='keras-succ-reg-wrapper',
    version='0.3.0',
    packages=find_packages(),
    url='https://github.com/CyberZHG/keras-succ-reg-wrapper',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='A wrapper that slows down the updates of trainable weights',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'keras',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
