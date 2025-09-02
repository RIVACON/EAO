import unittest
import numpy as np
import pandas as pd
import datetime as dt
import json
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '..'))
import eaopack as eao

class two_node_storage(unittest.TestCase):
    """ Tests for the extention of having a storage with two nodes (1) in (2) out. This allows
        for a simpler implementation of some specific situations """
    def test_simple_two_node_storage(self):
        """Simple test where first ten times price is zero and afterwards price is one, zero costs
           No difference to setup without nodes to be expected, since we are not using a portfolio
        """
        node1 = eao.assets.Node('N1')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.Storage('STORAGE', [node1, node2], start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10,\
             cap_in=1, cap_out=1, start_level=0, end_level=0, price='price')
        price = np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 10, 5)

    def test_portfolio_two_node_storage(self):
        """Simple test where first ten times price is zero and afterwards price is one, zero costs
           No difference to setup without nodes to be expected, since we are not using a portfolio
        """
        node1 = eao.assets.Node('N1')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.Storage('STORAGE', [node1, node2], start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10,\
             cap_in=1, cap_out=1, start_level=0, end_level=0, cost_in=.1)
        buy  = eao.assets.SimpleContract(name = 'buy', nodes = node1, min_cap=-10, max_cap=10, price = 'price')
        sell = eao.assets.SimpleContract(name = 'sell', nodes = node2, min_cap=-10, max_cap=10, price = 'price')        
        # cannot be used, since storage can only BRING volumes from node1 to node2
        portf = eao.portfolio.Portfolio([a, buy, sell])
        price = np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price, 'zero':np.zeros(timegrid.T), 'best':-np.ones(timegrid.T)}
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf = portf, op = op, res = res)
        self.assertAlmostEqual(res.value, 10-1, 5)

    def test_combination_two_node_storage(self):
        """Simple test where first ten times price is zero and afterwards price is one, zero costs
           No difference to setup without nodes to be expected, since we are not using a portfolio
        """
        node1 = eao.assets.Node('N1')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,5), freq = 'h')
        a1 = eao.assets.Storage('a1', [node1, node2], start=dt.datetime(2020,12,31,23), end=dt.date(2021,2,1),size=10,\
             cap_in=1, cap_out=1, start_level=0, end_level=0, cost_in=.1, block_size='2h')
        a2 = eao.assets.Storage('a2', [node1, node2], start=dt.datetime(2021,1,1,0), end=dt.date(2021,2,1),size=10,\
             cap_in=1, cap_out=1, start_level=0, end_level=0, cost_in=.1,block_size='2h')
        buy  = eao.assets.SimpleContract(name = 'buy', nodes = node1, min_cap=-10, max_cap=10, price = 'price')
        sell = eao.assets.SimpleContract(name = 'sell', nodes = node2, min_cap=-10, max_cap=10, price = 'price')        
        # cannot be used, since storage can only BRING volumes from node1 to node2
        portf = eao.portfolio.Portfolio([a1, a2, buy, sell])
        price = np.ones(timegrid.T)
        price[::3] = 0
        prices ={ 'price': price, 'zero':np.zeros(timegrid.T), 'best':-np.ones(timegrid.T)}
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf = portf, op = op, res = res, prices = prices)
        # eao.io.output_to_file(out, 'test.xlsx')
        self.assertAlmostEqual(res.value, 28.8, 5) # value from checking XLS output manually. all prices zero exploited



###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
