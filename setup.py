# -*- coding: utf-8 -*-
"""
Created on Sat May  9 23:43:56 2020

@author: leezi
"""

NAME = 'myfit'
DESCRIPTION = 'just test'
URL = 'https://github.com/lizihao1008'
EMAIL = 'leezihao_u@126.com'    # 你的邮箱
AUTHOR = 'Zihao Li'     # 你的名字

VERSION = '0.1'           # 项目版本号


def success():
    print('successful!')
    return

from setuptools import find_packages,setup

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,

    packages=find_packages(),



)

success()