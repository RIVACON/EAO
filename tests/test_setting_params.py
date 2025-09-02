import numpy as np
from numpy.core.shape_base import block
import pandas as pd
import datetime as dt
import json

import unittest   

from os.path import dirname, join
import os
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '..'))

import eaopack as eao

class test_setting_params(unittest.TestCase):

    def test_getting_and_setting_parameters(self):
        """ Unit test. Setting up a simple contract with random prices 
            and check that it buys full load at negative prices and opposite
        """

        ### define asset and get parameters
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.SimpleContract(name = 'SC', price = 'rand_price', nodes = node ,
                        min_cap= -10., max_cap=10., extra_costs=10, start = dt.date(2021,2,1))
        
        # to test: the timegrid is no parameter of the asset! This should get lost in the process
        a.set_timegrid(timegrid)
        # get parameter tree
        k,d = eao.io.get_params_tree(a)
        # d is a dictionary of all parameters (nested)
        # k is a list of parameter names (pointing downwards into nested objects)
        self.assertEqual(len(d), 15)
        self.assertEqual(len(k), 22)
        ### get parameter
        self.assertEqual(eao.io.get_param(a, 'name'), 'SC')
        self.assertEqual(eao.io.get_param(a,['nodes', 0, 'unit', 'volume']), 'MWh')
        ### set parameter
        a = eao.io.set_param(a, ['name'], 'test')
        self.assertEqual(a.name, 'test')
        a = eao.io.set_param(a, ['start', '__value__'], '2021-01-01')
        self.assertEqual(a.start, dt.date(2021,1,1))
        a = eao.io.set_param(a,['nodes', 0, 'unit', 'volume'], 'kWh')
        self.assertEqual(eao.io.get_param(a,['nodes', 0, 'unit', 'volume']), 'kWh')

    def test_setting_portfolio(self):
        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        node3 = eao.assets.Node('node_3')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=-5., start = dt.date(2021,1,10), end = dt.date(2021,1,20))
        a1a = eao.assets.SimpleContract(name = 'SC_1a', price = 'rand_price_1', nodes = node3 ,
                        min_cap= -20., max_cap= 20., start = dt.date(2021,1,10), end = dt.date(2021,1,20))

        #a1.set_timegrid(timegrid)
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -15., max_cap=-10.)#, extra_costs= 1.)
        #a2.set_timegrid(timegrid)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= 0., max_cap=100., extra_costs= 1.)
        a4 = eao.assets.Transport(name = 'Tr', costs_const= 5., nodes = [node2, node1],
                        min_cap= 0., max_cap=100.)

        portf = eao.portfolio.Portfolio([a1, a2, a3, a4,a1a])
        k,d = eao.io.get_params_tree(portf)
        p = eao.io.set_param(portf, ['assets', 0, 'nodes', 0, 'unit', 'flow'], 'kW')
        eao.io.get_param(portf, ['assets', 0, 'nodes', 0, 'unit', 'flow'])
        # original unchanged
        self.assertEqual(eao.io.get_param(portf, ['assets', 0, 'nodes', 0, 'unit', 'flow']), 'MW')
        # new changed
        self.assertEqual(eao.io.get_param(p, ['assets', 0, 'nodes', 0, 'unit', 'flow']), 'kW')
        # new changed
        pass


if __name__ == "__main__" :
    unittest.main()

