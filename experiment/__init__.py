import sys
import os.path as osp
__dir__ = osp.dirname(osp.abspath(__file__))
PYTHON_PATH = osp.join(__dir__, '..')
sys.path.insert(0, PYTHON_PATH)

import modules
import objects