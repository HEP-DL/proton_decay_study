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
    version='0.1.0',
    description="Looks for proton decay. USING NEURAL NETWORKS",
    long_description="TODO: Fill in",
    author="Kevin Wierman",
    author_email='kwierman@gmail.com',
    url='https://github.com/kwierman/proton_decay_study',
    packages=find_packages(),
    package_dir={'proton_decay_study':
                 'proton_decay_study'},
    entry_points={
        'console_scripts': [
            'proton_decay_study=proton_decay_study.cli:main'
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
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=dev_reqs
)
