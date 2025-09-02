import unittest
import numpy as np
import pandas as pd
import datetime as dt
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '..'))
import eaopack as eao


class TestTypedDict(unittest.TestCase):
    def test_typedDict(self):
        """ Unit test for tying typeddict
        """
        pass
        # show how typing works with specific types and classes
        # import typing
        # a = typing.get_type_hints(eao.assets.Contract.__init__)['min_take']
        # # <class 'eaopack.basic_classes.StartEndValueDict'>
        # # this gives uns the specific class for start/end/values
        # ###
        # typing.get_type_hints(a)
        #   {'start': typing.Sequence[date....datetime], 'end': typing.Sequence[float], 'value': typing.Sequence[float]}

        # So the idea of getting to understand eao logic in a gui, for example
        # would be to 
        # (1) check   typing.get_type_hints(eao.assets.Contract.__init__)
        #     to get the input variables
        # (2) check the details of specific classes such as node, unit, ...
        #     typing.get_type_hints(eao.basic_classes.StartEndValueDict)
        #     typing.get_type_hints(eao.basic_classes.Unit.__init__)
        #     typing.get_type_hints(eao.basic_classes.Node.__init__)
###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
