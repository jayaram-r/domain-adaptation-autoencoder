# -*- coding: utf-8 -*-
# little path hack to access module which is one directory up.
import sys
import os
LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)
