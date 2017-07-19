#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import setup, find_packages
from pip.req import parse_requirements
import pip


install_reqs = reqs = [str(ir.req) for ir in parse_requirements('requirements.txt',
    session=pip.download.PipSession())]
dev_reqs = [str(ir.req) for ir in parse_requirements('requirements_dev.txt',
    session=pip.download.PipSession())]

setup(
    name='proton_decay_study',
    version='0.2.0',
    description="Looks for proton decay. USING NEURAL NETWORKS",
    long_description="""
        Top-level code base for CNN study of LArTPC data for proton decay.

        This relies primarily on Kevlar and Keras with the Tensorflow backend.

        Kevlar provides the data interface consumed by the generators. Keras and
        Tensorflow provide the framework used to train and utilize the networks.

    """,
    author="Kevin Wierman",
    author_email='kevin.wierman@pnnl.gov',
    url='https://github.com/HEP-DL/proton_decay_study',
    packages=find_packages(),
    package_dir={'proton_decay_study':
                 'proton_decay_study'},
    entry_points={
        'console_scripts': [
            'test_file_input=proton_decay_study.cli:test_file_input',
            'test_threaded_files=proton_decay_study.cli:test_threaded_file_input',
            'train_kevnet=proton_decay_study.cli:train_kevnet',
            'train_widenet=proton_decay_study.cli:train_widenet',
            'make_kevnet_featuremap=proton_decay_study.cli:make_kevnet_featuremap'
        ]
    },
    include_package_data=True,
    install_requires=reqs,
    license="MIT license",
    zip_safe=False,
    keywords='proton_decay_study',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
    ],
    test_suite='tests',
    tests_require=dev_reqs
)
