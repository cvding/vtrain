#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time             : 2019/11/15 8:59 AM
# @Author           : zhifeng.ding
# @File             : setup.py
# @description      :

from distutils.core import setup
from setuptools import find_packages


setup(name='itrain',  # 包名
      version='1.0.0',  # 版本号
      description='my trainer for network',
      long_description='',
      author='zhifeng.ding',
      author_email='zhifeng.ding@vivo.com',
      url='https://gitlab.vmic.xyz/11110558/itrain',
      license='',
      install_requires=['pyaml'],
      dependency_links=[
          'https://pypi.tuna.tsinghua.edu.cn/simple'
      ],
      classifiers=[
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Natural Language :: Chinese (Simplified)',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.2',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Utilities'
      ],
      keywords='',
      packages=find_packages(where='src', exclude='', include=('*')),  # 必填
      package_dir={'': 'src'},  # 必填
      include_package_data=True,
      scripts=['scripts/vtrain'],
      )
