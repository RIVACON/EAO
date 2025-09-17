import unittest
import numpy as np
import pandas as pd
import datetime as dt
from copy import deepcopy

# in case eao is not installed
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '..'))
import eaopack as eao


class OptimizationTests(unittest.TestCase):

    def test_trivial_robust_optimization(self):

        """ Unit test. Setting up a simple portfolio to check restrictions on nodes and
            other basic functionality
        """

        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=20., start = dt.date(2021,1,10), end = dt.date(2021,1,20))
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -5., max_cap=10.)#, extra_costs= 1.)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= -10., max_cap=10., extra_costs= 1., start = dt.date(2021,1,10), end = dt.date(2021,1,25))
        a4 = eao.assets.Transport(name = 'Tr', costs_const= 5., nodes = [node1, node2],
                        min_cap= 0., max_cap=1.)
        prices ={'rand_price_1': np.ones(timegrid.T)*1.,
                'rand_price_2': np.ones(timegrid.T)*10.,
                }
        
        portf = eao.portfolio.Portfolio([a1, a2, a3, a4])
        ###############################################   std optimization

        op_std  = portf.setup_optim_problem(prices, timegrid)
        res_std = op_std.optimize()

        ###############################################   robust optimization - twice the original prices as samples
        trivial_sample = [op_std.c, op_std.c]
        res_check  = op_std.optimize(target = 'robust', samples = trivial_sample)
        
        self.assertAlmostEqual(res_std.value, res_check.value, 5)


    def test_trivial_slp(self):
        """ Unit test. Setting up a simple portfolio to check restrictions on nodes and
            other basic functionality
        """
        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')

        Start = dt.date(2021,1,1)
        End   = dt.date(2021,1,10)

        start_future = dt.date(2021,1,4)

        timegrid = eao.assets.Timegrid(Start, End, freq = 'd')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'price1', nodes = node1 ,
                        min_cap= -20., max_cap=20.)
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'price1', nodes = node1 ,
                        min_cap= -5., max_cap=10.)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'price2', nodes = node2 ,
                        min_cap= -10., max_cap=10., extra_costs= 1.)
        a4 = eao.assets.Transport(name = 'Tr', costs_const= 0., nodes = [node1, node2],
                        min_cap= 0., max_cap=1.)

        # starting point with full set of prices across the grid
        price1 = {'start' : Start, 'values': 1.}
        price2 = {'start' : Start, 'values': 20.}
        
        # across all times
        fullSet1 = timegrid.values_to_grid(price1)
        fullSet2 = timegrid.values_to_grid(price2)
        prices_full = {'price1': fullSet1, 'price2': fullSet2}
        portf  = eao.portfolio.Portfolio([a1, a2, a3, a4])
        op     = portf.setup_optim_problem(prices_full, timegrid)
        res = op.optimize()
        op_slp = eao.stoch_lin_prog.make_slp(portf = portf, 
                                             optim_problem= deepcopy(op),
                                             timegrid=timegrid,
                                             start_future = start_future, 
                                             samples = [prices_full]*3) 
        # check optimization
        res_slp = op_slp.optimize()
        # both must give same result, since I also gave the SLP the identical price samples
        self.assertAlmostEqual(res.value, res_slp.value, 5)

    def test_fixing_results(self):
        """ Test fixing results - for a time window. E.g. when looping through present/future in SLP """
        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        Start = dt.date(2021,1,1) 
        End   = dt.date(2021,2,1)
        fix_date = Start + dt.timedelta(days = 10)

        timegrid = eao.assets.Timegrid(Start, End, freq = 'd')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=20., start = dt.date(2021,1,10), end = dt.date(2021,1,20))
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_2', nodes = node1 ,
                        min_cap= -5., max_cap=10.)#, extra_costs= 1.)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= -1., max_cap=10., extra_costs= 1.)
        a5 = eao.assets.Storage('storage', nodes = node1, \
             start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
             cap_in=1.0/24.0, cap_out=1.0/24.0, start_level=5, end_level=5)
        pricesA ={'rand_price_1': np.sin(np.linspace(0,10,timegrid.T)),
                'rand_price_2': np.cos(np.linspace(0,10,timegrid.T))}
        pricesB ={'rand_price_2': np.sin(np.linspace(0,10,timegrid.T)),
                'rand_price_1': np.cos(np.linspace(0,10,timegrid.T))}
        
        portf = eao.portfolio.Portfolio([a1, a2, a3, a5])
        ### original
        opA    = portf.setup_optim_problem(pricesA, timegrid)
        resA = opA.optimize()
        ### not fixed, new situation
        opB    = portf.setup_optim_problem(pricesB, timegrid)
        resB = opB.optimize()
        assert abs(resA.value - resB.value) >= 1e-5 # would result really be different?
        ### now fixed, but prices would result in different solution
        fix_time_window = {'I': fix_date, 'x': resA.x}
        opC    = portf.setup_optim_problem(pricesB, timegrid, fix_time_window=fix_time_window)
        I = (timegrid.timepoints<= pd.Timestamp(fix_date))
        I = opC.mapping['time_step'].isin(timegrid.I[I])
        resC = opC.optimize()                           
        self.assertAlmostEqual(sum(abs((resA.x[I] - resC.x[I]))), 0., 5)  

    def test_flag_MIP(self):
        """ Unit test. simple optim prob - check "is MIP" flag
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,2), freq = 'h')
        a = eao.assets.Storage('STORAGE', node, 
                               size=5,
                               cap_in=1,
                               cap_out=1, 
                               start_level=0, 
                               end_level=0, 
                               price='price',
                               eff_in=.8, 
                               eff_out=0.9,
                               no_simult_in_out = True)
        price = np.ones([timegrid.T])
        price[:10] = 0
        price[8] = 5
        price[3:5] = 0
        price[18:20] = 20

        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        ### check flag - with no_simult_in_out= True should be MIP
        self.assertTrue(op.is_MIP)
        ### check flag - with no_simult_in_out= False should be LP
        a.no_simult_in_out = False
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        self.assertTrue(not op.is_MIP)

class SplitOptimizationTests(unittest.TestCase):
    def test_same_same(self):
        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        Start = dt.date(2021,2,10) 
        End   = dt.date(2021,3,12)
        timegrid = eao.assets.Timegrid(Start, End, freq = 'h')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=20., start = dt.date(2021,2,10), end = dt.date(2021,3,20), wacc=0.2)
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_2', nodes = node1 ,
                        min_cap= -5., max_cap=10., wacc=0.)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= -1., max_cap=10., extra_costs= 1.)
        a5 = eao.assets.Storage('storage', nodes = node1, \
             start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
             cap_in=1.0/24.0, cap_out=1.0/24.0, start_level=5, end_level=5,
             block_size='d')
        pricesA ={'rand_price_1': np.sin(np.linspace(0,10,timegrid.T)),
                'rand_price_2': np.cos(np.linspace(0,10,timegrid.T))}

        portf = eao.portfolio.Portfolio([a1, a2, a3, a5])
        ### original
        opA    = portf.setup_optim_problem(pricesA, timegrid)
        resA = opA.optimize()
        outA = eao.io.extract_output(portf, opA, resA)
        ### split_optim
        opB    = portf.setup_split_optim_problem(pricesA, timegrid, interval_size='d')
        resB   = opB.optimize()
        outB = eao.io.extract_output(portf, opB, resB)
        # all results must be equal        
        self.assertAlmostEqual(resA.value, resB.value, 4)
        self.assertTrue(all(abs(outA['dispatch']-outB['dispatch']).sum()<1e-5))
        self.assertTrue(all(abs(outA['prices']-outB['prices']).sum()<1e-3))
###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
