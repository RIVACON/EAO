import unittest
import numpy as np
import pandas as pd
import datetime as dt
import json
from os.path import dirname, join
import sys

mypath = dirname(__file__)
sys.path.append(join(mypath, ".."))

import eaopack as eao


class mapping_logic(unittest.TestCase):
    def test_setting_up_mapping(self):
        """ set up empty basic mapping"""
        map = eao.basic_classes._mapping_create_empty()
        pass

###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == "__main__":
    unittest.main()
