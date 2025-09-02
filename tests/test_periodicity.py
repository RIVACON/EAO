import unittest
import numpy as np
import pandas as pd
import datetime as dt
import json
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '..'))
pd.set_option("display.max_rows", 10000, "display.max_columns", 10000)
import eaopack as eao

class PeriodicityTests(unittest.TestCase):

    def test_simple_contract(self):
        """ Unit test. Setting up a simple contract with random prices
            and check that it buys full load at negative prices and opposite
            --- with extra costs (in and out dispatch)
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'h')
        a = eao.assets.SimpleContract(name = 'SC', price = 'price', nodes = node ,
                                      min_cap= -10., max_cap=+10., 
                                      start =dt.date(2021,1,3), end = dt.date(2021,1,25),
                                      periodicity= 'd',
                                      periodicity_duration = None)
        b = eao.assets.SimpleContract(name = 'm',  nodes = node ,
                                      min_cap= -10., max_cap=+10.)

        portf = eao.portfolio.Portfolio([a,b])
        prices ={'price': np.sin(30*np.linspace(0,10,timegrid.T))}
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        ### checks
        d = out['dispatch']
        # outside scope
        assert all(d.loc[(d.index<pd.Timestamp(dt.date(2021,1,3)))|(d.index>=pd.Timestamp(dt.date(2021,1,25))), 'SC'] == 0)
        assert all(d.loc[(d.index<pd.Timestamp(dt.date(2021,1,3)))|(d.index>=pd.Timestamp(dt.date(2021,1,25))), 'm'] == 0)
        # all hours equal
        d = d[(d.index>=pd.Timestamp(dt.date(2021,1,3)))&(d.index<pd.Timestamp(dt.date(2021,1,25)))]
        for h in range(0,24):
            assert all(d.loc[d.index.hour == h, 'SC'] == d.loc[d.index.hour == h, 'SC'][0])

    def test_periodic_contract_max_capa(self):
        node = eao.assets.Node('testNode')
        Start = dt.date(2021,1,1)
        End   = dt.date(2021,1,10)
        timegrid = eao.assets.Timegrid(Start, End, freq = 'h')
        np.random.seed(2709)
        # capacities    
        restr_times = pd.date_range(Start, End, freq = 'h', inclusive = 'left')
        min_cap = {}
        min_cap['start']  = restr_times.to_list()
        min_cap['values'] = np.random.rand(len(min_cap['start'] ))

        # simple daily
        a = eao.assets.Contract(name = 'SC', price = 'rand_price', nodes = node ,
                        min_cap= min_cap, max_cap=min_cap,
                        periodicity = 'd')

        b = eao.assets.SimpleContract(name = 'm',  nodes = node ,
                                      min_cap= -10., max_cap=+10.)

        prices = {'rand_price': np.sin(np.linspace(0,50,timegrid.T))}
        portf = eao.portfolio.Portfolio([a,b])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        disp = out['dispatch']['SC']
        # for each hour expect the mean of the max capa - since we average l & u
        for hh in range(0,24):      
            I = disp.index.hour == hh
            for aa in disp[I]:
                self.assertAlmostEqual(min_cap['values'][I].mean(),aa, 5)
        pass        
        # cash flow (1) value and details
        dcf = out['DCF']['SC']
        self.assertAlmostEqual(dcf.sum(), res.value, 5)
        # cash flow (2) derived com price ... 
        pp = prices['rand_price']
        self.assertAlmostEqual(-(pp*disp).sum(), res.value, 5)

    def test_period_contract_min_take(self):
        node = eao.assets.Node('testNode')
        Start = dt.date(2021,1,1)
        End   = dt.date(2021,1,10)
        startA = dt.date(2021,1,3)
        timegrid = eao.assets.Timegrid(Start, End, freq = 'h')
        # capacities    
        restr_times = pd.date_range(Start, End, freq = 'd', inclusive = 'left')
        min_cap = {}
        min_cap['start']  = restr_times.to_list()
        min_cap['end']    = (restr_times + dt.timedelta(days = 1)).to_list()
        min_cap['values'] = np.zeros(len(min_cap['start'] ))
        max_cap = min_cap.copy()
        max_cap['values'] = np.ones(len(min_cap['start'] ))*10.
        # min and max take
        #restr_times = pd.date_range(Start, End, freq = '4h', inclusive = 'left')
        min_take = {}
        min_take['start']  = Start
        min_take['end']    = End
        min_take['values'] = 20
        #max_take = min_take.copy()
        max_take = None
        a = eao.assets.Contract(name = 'SC', price = 'rand_price', nodes = node , start=startA,
                        min_cap= min_cap, max_cap=max_cap, min_take=min_take, max_take = max_take,
                        periodicity = '4h')
        b = eao.assets.SimpleContract(name = 'm',  nodes = node ,
                                min_cap= -1000., max_cap=+1000.)
        portf = eao.portfolio.Portfolio([a,b])
        prices = {'rand_price': np.ones(timegrid.T)}
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        #periodicity must not affect max_take
        sdisp = out['dispatch']['SC'].sum() * (End-Start)/(End-startA)  
        self.assertAlmostEqual(sdisp, 20., 5)

    def test_period_transport(self):
        """ make transport periodic - should have same effect
        """
        node1 = eao.assets.Node('n1')
        node2 = eao.assets.Node('n2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'h')
        a = eao.assets.SimpleContract(name = 'SC', price = 'price', nodes = node1 ,
                                      min_cap= -10., max_cap=+10., 
                                      start =dt.date(2021,1,3), end = dt.date(2021,1,25),
                                      )# periodicity = 'd')
        b = eao.assets.SimpleContract(name = 'm',  nodes = node2 ,
                                      min_cap= -10., max_cap=+10.)
        t = eao.assets.Transport(name = 't', min_cap = -1000, max_cap= 1000.,  nodes=[node1, node2],
                        periodicity = 'd',start =dt.date(2021,1,3), end = dt.date(2021,1,25))
        portf = eao.portfolio.Portfolio([a,b,t])
        prices ={'price': np.sin(30*np.linspace(0,10,timegrid.T))}
        #opt = t.setup_optim_problem(prices, timegrid=timegrid)
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        ### checks
        d = out['dispatch']
        # outside scope
        assert all(d.loc[(d.index<pd.Timestamp(dt.date(2021,1,3)))|(d.index>=pd.Timestamp(dt.date(2021,1,25))), 'SC (n1)'] == 0)
        assert all(d.loc[(d.index<pd.Timestamp(dt.date(2021,1,3)))|(d.index>=pd.Timestamp(dt.date(2021,1,25))), 'm (n2)'] == 0)
        # all hours equal
        d = d[(d.index>=pd.Timestamp(dt.date(2021,1,3)))&(d.index<pd.Timestamp(dt.date(2021,1,25)))]
        for h in range(0,24):
            self.assertAlmostEqual(abs(d.loc[d.index.hour == h, 'SC (n1)'] - d.loc[d.index.hour == h, 'SC (n1)'][0]).sum(), 0, 4)

    def test_period_storage(self):
        """ Periodization. Unit test. Setting up a simple contract with random prices 
            and check that it buys full load at negative prices and opposite
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.Storage(name = 'st', nodes = node ,
                               cap_in=1, cap_out=1, size = 5,
                               periodicity = '4d')
        b = eao.assets.SimpleContract(name = 'm',  nodes = node , price = 'price',
                                      min_cap= -10., max_cap=+10.)
        prices ={'price': np.sin(30*np.linspace(0,10,timegrid.T))}
        portf = eao.portfolio.Portfolio([a,b])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        d = out['dispatch']
        for ii in range(0,4):
            dd = timegrid.timepoints[ii::4]
            self.assertAlmostEqual(abs(d.loc[dd, 'st'] - d.loc[dd, 'st'][0]).sum(), 0, 4)


###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
