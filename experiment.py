# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 21:02:58 2020

@author: Rajiv
"""

import numpy as np

a = np.arange(1,11).reshape(1,10)
print(a)

b = np.arange(1,3).reshape(1,2)
print(b)

print(a*b)