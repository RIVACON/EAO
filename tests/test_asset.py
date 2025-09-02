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

class SimpleContractTest(unittest.TestCase):
    def test_optimization(self):
        """ Unit test. Setting up a simple contract with random prices 
            and check that it buys full load at negative prices and opposite
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.SimpleContract(name = 'SC', price = 'rand_price', nodes = node ,
                        min_cap= -10., max_cap=10.)
        #a.set_timegrid(timegrid)
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        # check for this case if result makes sense. Easy: are signs correct?
        # buy for negative price foll load, sell if opposite
        # check = all(np.sign(np.around(res.x, decimals = 3)) != np.sign(op.c))
        x = np.around(res.x, decimals = 3) # round
        check =     all(x[np.sign(op.c) == -1] == op.u[np.sign(op.c) == -1]) \
                and all(x[np.sign(op.c) == 1]  == op.l[np.sign(op.c) == 1])
        tot_dcf = np.around((a.dcf(op, res)).sum(), decimals = 3) # asset dcf, calculated independently
        check = check and (tot_dcf == np.around(res.value , decimals = 3))
        self.assertTrue(check)

    def test_optimization_ec(self):
        """ Unit test. Setting up a simple contract with random prices 
            and check that it buys full load at negative prices and opposite
            --- with extra costs (in and out dispatch)
        """
        node = eao.assets.Node('testNode')
        unit = eao.assets.Unit
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.SimpleContract(name = 'SC', price = 'rand_price', nodes = node ,
                        min_cap= -20., max_cap=+20., extra_costs= 0.3, start =dt.date(2021,1,3), end = dt.date(2021,1,25) )
        a.set_timegrid(timegrid)    
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op = a.setup_optim_problem(prices)
        res = op.optimize()
        # check for this case if result makes sense. Easy: are signs correct?
        # buy for negative price foll load, sell if opposite
        # check = all(np.sign(np.around(res.x, decimals = 3)) != np.sign(op.c))
        x = np.around(res.x, decimals = 3) # round
        check =     all(x[np.sign(op.c) == -1] == op.u[np.sign(op.c) == -1]) \
                and all(x[np.sign(op.c) == 1]  == op.l[np.sign(op.c) == 1])

        tot_dcf = np.around((a.dcf(op, res)).sum(), decimals = 3) # asset dcf, calculated independently
        check = check and (tot_dcf == np.around(res.value , decimals = 3))
        self.assertTrue(check)
        # test variable naming
        self.assertTrue(op.mapping.loc[0].var_name == 'disp_in')
        self.assertTrue(op.mapping.loc[22].var_name == 'disp_out')


class StorageTest(unittest.TestCase):
    def test_optim_trivial(self):
        """Simple test where first ten times price is zero and afterwards price is one, zero costs
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10,\
             cap_in=1, cap_out=1, start_level=0, end_level=0, price='price')
        price = np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 10, 5)
        print(res)

               

    def test_optim_trivial_blocks(self):
        """Simple test where first ten times price is zero and afterwards price is one, zero costs
        """
        #### case 1: with days (same block length)
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        #a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
        #                       cap_in=1, cap_out=1, start_level=0, end_level=0, block_size= 7*24 ,price='price')
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
                               cap_in=1, cap_out=1, start_level=0, end_level=0, block_size= '7d' ,price='price')

        # in timegrid, main_time_unit is std. set to 'h'- means to obtain a day it needs be 24
        price = np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 10, 5)
        # every 10th fill level must be zero
        self.assertAlmostEqual(res.x.cumsum()[6], 0, 5)
        self.assertAlmostEqual(res.x.cumsum()[13], 0, 5)
        self.assertAlmostEqual(res.x.cumsum()[20], 0, 5)
        self.assertAlmostEqual(res.x.cumsum()[27], 0, 5)        

        #### case 2: with months (varying block length)
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,7,1), freq = '1d')
        #a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
        #                       cap_in=1, cap_out=1, start_level=0, end_level=0, block_size= 7*24 ,price='price')
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,12,2),size=10, \
                               cap_in=1, cap_out=1, start_level=0, end_level=0, block_size= 'MS' ,price='price')

        # in timegrid, main_time_unit is std. set to 'h'- means to obtain a day it needs be 24
        price = np.sin(np.linspace(0,20,timegrid.T))
        price[:10] = 0
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        # every 10th fill level must be zero
        i = 0
        for myd in timegrid.timepoints:
            if myd.day == 1:
                if i !=0:
                 self.assertAlmostEqual(res.x.cumsum()[i-1], 0, 5)
            i +=1


    def test_optim_trivial_costs(self):
        """Simple test where first ten times price is zero and afterwards price is one but with different costs
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        price = np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price}

        # cost_in = 0.1
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
            cap_in=1, cap_out=1, start_level=0, end_level=0, price='price', cost_in=0.1)
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 9.0, 5)

        # cost_out = 0.1
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, cap_in=1, cap_out=1, start_level=0, end_level=0, price='price', cost_in=0.0, cost_out=0.1)
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 9.0, 5)
        self.assertTrue(op.mapping.loc[33].var_name == 'disp_out')
        self.assertTrue(op.mapping.loc[2].var_name == 'disp_in')

    def test_optim_nonzero_capacity(self):
        """Simple test where first ten times price is zero and afterwards price is one but with capacity=1.0/24.0
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        price = np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price}

        # cost_in = 0.1
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), \
            end=dt.date(2021,2,1),size=10, cap_in=1.0/24.0, cap_out=1.0/24.0, \
                start_level=0, end_level=0,cost_in = 0.2,  price='price')
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 10-10*0.2, 5)
        dispatch = res.x[:31]+res.x[31:]
        for x in dispatch:
            self.assertLessEqual(np.abs(x), 1.0)

    def test_trivial_storage_with_inflow(self):
        """ storage with zero storage size, but given inflow """

        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,2), freq = 'd', main_time_unit='d')
        prices ={ 'price': np.ones([timegrid.T])}

        a = eao.assets.Storage('STORAGE', node, \
                                start=dt.date(2021,1,1), end=dt.date(2021,2,2),\
                                size=0, cap_in=10, cap_out=10, inflow= 1, cost_in=1, cost_out=1,\
                                start_level=0, end_level=0, price='price')
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        T = timegrid.T
        d = res.x[0:T]+res.x[T:]
        # result should be exactly the inflow
        self.assertAlmostEqual((d-1).sum(), 0, 5)
       
    def test_store_cost_with_cap_costs(self):
        """ Test on storage costs for stored volume - with efficiency
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,2), freq = 'h')
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10,\
             cap_in=5, cap_out=5, start_level=0, end_level=0, price='price',
             cost_store= .5, eff_in= 1, cost_in=1, cost_out=2)
        price = 1e5*np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        ### charge / discharge should be close to each other to avoid charged storage
        for ii in range(0,8):
            self.assertAlmostEqual(res.x[ii], 0, 3)
        for ii in range(8,10):
            self.assertAlmostEqual(res.x[ii], -5, 3)
        for ii in range(34,36):
            self.assertAlmostEqual(res.x[ii], 5, 3)
        for ii in range(36,48):
            self.assertAlmostEqual(res.x[ii], 0, 3)                   
        # total value: earnung 10+1e5, costs for storage .5 per MWh in storage     
        self.assertAlmostEqual(res.value, 1e6-(5+10+5)*.5-10*1-10*2, 3)
        print(res)

    def test_store_cost_eff(self):
        """ Test on storage costs for stored volume - with efficiency
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,2), freq = 'h')
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10,\
             cap_in=5, cap_out=5, start_level=0, end_level=0, price='price',
             cost_store= .5, eff_in= .8, cost_in=1, cost_out=1)
        price = 1e5*np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        ### charge / discharge should be close to each other to avoid charged storage
        for ii in range(0,7):
            self.assertAlmostEqual(res.x[ii], 0, 3)
        # lost must be met by dispatch before full loading
        self.assertAlmostEqual(res.x[7], -(10*.2)/.8, 3)
        for ii in range(8,10):
            self.assertAlmostEqual(res.x[ii], -5, 3)
        for ii in range(34,36):
            self.assertAlmostEqual(res.x[ii], 5, 3)
        for ii in range(36,48):
            self.assertAlmostEqual(res.x[ii], 0, 3)                   
        # total value: earnung 10+1e5, costs for storage .5 per MWh in storage     
        self.assertAlmostEqual(1e6-res.value, (+10+10/.8)+(((2.5+7.5+12.5)*.8 + 5)*.5), 3)

class TransportTest(unittest.TestCase):
    def test_transport(self):
        """ Unit test. Setting up transport with random costs
        """
        node1 = eao.assets.Node('N1')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.Transport(name = 'Tr', costs_const= 5., costs_time_series= 'rand_price', nodes = [node1, node2],
                        min_cap= 0., max_cap=10.)
        #a.set_timegrid(timegrid)
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        # check for this case if result makes sense. Easy: are signs correct?
        # buy for negative price foll load, sell if opposite
        # check = all(np.sign(np.around(res.x, decimals = 3)) != np.sign(op.c))
        x = np.around(res.x, decimals = 3) # round
        check = sum(abs(x)) <= 1e-5 
        tot_dcf = np.around((a.dcf(op, res)).sum(), decimals = 3) # asset dcf, calculated independently
        check = check and (tot_dcf == np.around(res.value , decimals = 3))
        self.assertTrue(check)
        
    def test_extended_transport(self):
        """ needed to define min/max take on transport"""
        node1a = eao.assets.Node('N1a')
        node1b = eao.assets.Node('N1b')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,11), freq = 'd', main_time_unit='d')
        prices ={'buy': np.zeros(timegrid.T), 'sell': 1*np.ones(timegrid.T)}
        max_take = {'start':dt.date(2021,1,1),
                    'end':dt.date(2021,1,11),
                    'values' : 2.5}
        transa = eao.assets.ExtendedTransport(name = 'TrA', nodes = [node1a, node2],
                        min_cap= 0., max_cap=10., max_take=max_take)
        transb = eao.assets.ExtendedTransport(name = 'TrB', nodes = [node1b, node2],
                        min_cap= 0., max_cap=10., max_take=max_take)                        
        buya  = eao.assets.SimpleContract(name = 'buya', price = 'buy', max_cap = 2, nodes = node1a  )
        buyb  = eao.assets.SimpleContract(name = 'buyb', price = 'buy', max_cap = 2, nodes = node1b  )
        sell = eao.assets.SimpleContract(name = 'sell', price = 'sell', min_cap = -1, nodes = node2  )

        portf = eao.portfolio.Portfolio([transa, transb, buya, buyb, sell])
        op = portf.setup_optim_problem(prices, timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value,5., 5)

    def test_extended_transport_max_min_take(self):
        """ needed to define min/max take on transport"""
        node1 = eao.assets.Node('N1')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,20), freq = 'd', main_time_unit='d')
        prices ={'buy': np.zeros(timegrid.T), 'sell': 1*np.ones(timegrid.T)}
        max_take = {'start':[dt.date(2021,1,1), dt.date(2021,1,2), dt.date(2021,1,11)],
                    'end':  [dt.date(2021,1,2), dt.date(2021,1,11), dt.date(2021,2,11)],
                    'values' : [0,2.2,0]}
        trans = eao.assets.ExtendedTransport(name = 'TrB', nodes = [node1, node2],
                        min_cap= 0., max_cap=10., max_take=max_take, min_take=max_take)                        

        buy  = eao.assets.SimpleContract(name = 'buya', price = 'buy', max_cap = 2, nodes = node1  )
        sell = eao.assets.SimpleContract(name = 'sell', price = 'sell', min_cap = -1, nodes = node2  )

        portf = eao.portfolio.Portfolio([trans, buy, sell])

        ## check max_take
        op = portf.setup_optim_problem(prices, timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf=portf, op = op, res = res)
        self.assertAlmostEqual(out['dispatch'].loc[pd.Timestamp(2021,1,1):pd.Timestamp(2021,1,1),'buya (N1)'].sum(),0., 5)
        self.assertAlmostEqual(out['dispatch'].loc[pd.Timestamp(2021,1,2):pd.Timestamp(2021,1,10),'buya (N1)'].sum(),2.2, 5)        
        self.assertAlmostEqual(out['dispatch'].loc[pd.Timestamp(2021,1,11):pd.Timestamp(2021,2,10),'buya (N1)'].sum(),0, 5)          
        self.assertAlmostEqual(res.value, 2.2, 5)
        ## check min_take
        prices ={'buy': np.zeros(timegrid.T), 'sell': -1*np.ones(timegrid.T)}
        op = portf.setup_optim_problem(prices, timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf=portf, op = op, res = res)
        self.assertAlmostEqual(out['dispatch'].loc[pd.Timestamp(2021,1,1):pd.Timestamp(2021,1,1),'buya (N1)'].sum(),0., 5)
        self.assertAlmostEqual(out['dispatch'].loc[pd.Timestamp(2021,1,2):pd.Timestamp(2021,1,10),'buya (N1)'].sum(),2.2, 5)        
        self.assertAlmostEqual(out['dispatch'].loc[pd.Timestamp(2021,1,11):pd.Timestamp(2021,2,10),'buya (N1)'].sum(),0, 5)          
        self.assertAlmostEqual(res.value, -2.2, 5)

    def test_transport_efficiency(self):
        """ check efficiency in transport """
        node1a = eao.assets.Node('N1a')
        node1b = eao.assets.Node('N1b')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,11), freq = 'd', main_time_unit='d')
        prices ={'buy': np.ones(timegrid.T), 'sell': 10.*np.ones(timegrid.T)}
        trans = eao.assets.Transport(name = 'TrA', nodes = [node1a, node2],
                        min_cap= 0., max_cap=10., efficiency = 0.95)
        buy  = eao.assets.SimpleContract(name = 'buya', price = 'buy', max_cap = 1, nodes = node1a  )
        sell = eao.assets.SimpleContract(name = 'sell', price = 'sell', min_cap = -1, nodes = node2  )

        portf = eao.portfolio.Portfolio([trans, buy, sell])
        op = portf.setup_optim_problem(prices, timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 10*(9.5-1), 5) # buy one (at price 1), get 0.95 out (at price 10) for each day

class ContractTest(unittest.TestCase):
    def test_max_cap_vector(self):
        node = eao.assets.Node('testNode')
        Start = dt.date(2021,1,1)
        End   = dt.date(2021,1,10)
        timegrid = eao.assets.Timegrid(Start, End, freq = 'h')

        # capacities    
        restr_times = pd.date_range(Start, End, freq = 'd', inclusive = 'left')
        min_cap = {}
        min_cap['start']  = restr_times.to_list()
        min_cap['end']    = (restr_times + dt.timedelta(days = 1)).to_list()
        min_cap['values'] = np.random.rand(len(min_cap['start'] ))
        max_cap = min_cap.copy()

        a = eao.assets.Contract(name = 'SC', price = 'rand_price', nodes = node ,
                        min_cap= min_cap, max_cap=min_cap)
        
        prices = {'rand_price': -np.ones(timegrid.T)}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual((res.x - timegrid.values_to_grid(max_cap)).sum(),0., 5)
        # check serialization (new class...)
        s = eao.serialization.to_json(a)

    def test_contract_min_take(self):
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
                        min_cap= min_cap, max_cap=max_cap, min_take=min_take, max_take = max_take)
        
        prices = {'rand_price': np.ones(timegrid.T)}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        sdisp = res.x.sum() * (End-Start)/(End-startA) - 20.        
        self.assertAlmostEqual(sdisp, 0., 5)

    def test_max_cap_vector_from_pricesDF(self):
        node = eao.assets.Node('testNode')
        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 10)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        # capacities
        min_cap = np.random.rand(timegrid.T)
        prices = {'rand_price': -np.ones(timegrid.T), 'cap': min_cap}

        a = eao.assets.Contract(name='SC', price='rand_price', nodes=node,
                                min_cap='cap', max_cap='cap')

        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(abs(min_cap - res.x).sum(), 0., 5)

class MultiCommodity(unittest.TestCase):

    def test_predefined_multicommodity(self):
        """ test to reproduce the results in the sample "capture_heat_portfolio.py """
        portf = eao.serialization.load_from_json(file_name = join(mypath,'test_portf_multi_commodity.JSON'))
        prices = pd.read_csv(join(mypath, '2020_price_sample.csv'))
        # cast to timegrid
        prices = {'price': portf.timegrid.values_to_grid({'start': pd.to_datetime(prices['start'].values), 'values': prices['price'].values})}
        op = portf.setup_optim_problem(prices)
        res = op.optimize()
        # checking against known value --> no change
        self.assertAlmostEqual(res.value, 3307.322231803014, 4)
        out = eao.io.extract_output(portf, op, res, prices)
        self.assertAlmostEqual(res.value, out['DCF'].sum().sum()) # check detailed - asset-wise DCF is equal to LP value
        # values of heat demand
        out['dispatch']['heat_demand (heat)'].sum()
        heat_res = out['dispatch']['heat_demand (heat)'].values
        self.assertAlmostEqual(heat_res.sum(), -287.6800669895399, 4)
        self.assertAlmostEqual(heat_res[-1], -0.7309221113481956, 4)
        self.assertAlmostEqual(heat_res[0], 0, 4)        
        self.assertAlmostEqual(heat_res[71], -0.05461761740138421, 4)        
        
class ScaledAsset(unittest.TestCase):
    def test_scaled_asset(self):
        node = eao.assets.Node('testNode')
        Start = dt.date(2021,1,1)
        End   = dt.date(2021,1,10)
        timegrid = eao.assets.Timegrid(Start, End, freq = 'h')


        a = eao.assets.SimpleContract(name = 'market',price  = 'spot', min_cap= -10, max_cap= 10, nodes = node )
        b = eao.assets.Storage(name='battery', nodes = node, cap_in= 1, cap_out= 1, size=4)

        scaled_b = eao.assets.ScaledAsset(name = 'scaled_b', base_asset = b, max_scale= 1, fix_costs= 0.)

        prices = {'spot': 10*np.sin(np.linspace(0,10, timegrid.T)) }
        # standard case
        portf_std = eao.portfolio.Portfolio([a,b])
        op_std  = portf_std.setup_optim_problem(prices, timegrid=timegrid)
        res_std = op_std.optimize()
        # standard case
        portf = eao.portfolio.Portfolio([a,scaled_b])
        op  = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf=portf, op = op, res = res)
        # zero costs and limit at original size -- same result
        self.assertAlmostEqual(res_std.value, res.value, 4)

    def test_scaled_transport(self):
        """ test from a n actual analysis with a more complex portfolio """
        S = dt.date(2023,1,1)
        E = dt.date(2023,1,2)
        timegrid = eao.assets.Timegrid(S, E, freq = 'h')
        input_ts = {'price': np.asanyarray([1,2,-1,2,1,4,2,4,5,115,7,3,1,2,-1,2,1,4,2,4,5,115,7,3]), 'cons': -2*np.asanyarray([0,0,1,1,1,1,1,4,1,2,1,1,1,1,2,1,2,1,2,2,2,5,7,3])}
        behind_meter = eao.assets.Node('behind meter')
        front_of_meter = eao.assets.Node('front of meter')

        consumption = eao.assets.SimpleContract(name       = 'consumption', 
                                                nodes      = behind_meter,
                                                min_cap    = 'cons',
                                                max_cap    = 'cons')

        supply  = eao.assets.SimpleContract(    name = 'supply', 
                                                nodes       = front_of_meter, 
                                                price       = 'price',
                                                min_cap     = 0,
                                                max_cap     = 100)
        grid_consumption_normed = eao.assets.Transport(name   = 'grid_in_normed',  # 
                                                    nodes  = [front_of_meter, behind_meter],
                                                    #costs_const = grid_fees['var'], # variable grid fees
                                                    min_cap= 0,
                                                    max_cap= 1) # normed capacity (to 1 MW)
        grid_consumption      = eao.assets.ScaledAsset(name       = 'grid_in',
                                                    base_asset = grid_consumption_normed,
                                                    max_scale  = 1000,
                                                    fix_costs  = 100) # yearly fix capacity costs (here scaled to hourly as main time unit)
        battery     = eao.assets.Storage(       name       = 'battery',
                                                nodes      = behind_meter,
                                                cap_in     = 1,
                                                cap_out    = 1,                                        
                                                size       = 2)
        portf = eao.portfolio.Portfolio([supply, consumption,  grid_consumption, battery])

        op  = portf.setup_optim_problem(prices = input_ts, timegrid = timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, input_ts)
        self.assertAlmostEqual(out['special'].loc[0,'costs'], 31200, 1) # simply functional test

class DiscountRate(unittest.TestCase):
    def test_discount_simple_contract(self):
        """ Unit test. Simple contract with discount rate
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(1999,1,1), dt.date(2000,1,1), freq = 'd')
        a = eao.assets.SimpleContract(name = 'SC', extra_costs=-1, price = 'pr', nodes = node ,
                        min_cap= 0, max_cap=10., wacc = 0.1)
        prices = {'pr':0.2*np.ones([timegrid.T])}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        timegrid.set_wacc(0.1)
        self.assertAlmostEqual(timegrid.discount_factors[-1], 1./1.1, 4)
        check = 10*(1-0.2)*timegrid.discount_factors.sum()*24
        self.assertAlmostEqual(check, res.value, 2)

    def test_store_cost_with_cap_costs_discount(self):
        """ Test on storage costs for stored volume - with efficiency
        """
        ### (A) without discount
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,25), freq = 'd', main_time_unit='d')
        a = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10,\
             cap_in=5, cap_out=5, start_level=0, end_level=0, price='price',
             cost_store= .5, eff_in= 1, cost_in=1, cost_out=2, wacc = 0.)
        price = 1e3*np.ones([timegrid.T])
        price[:10] = 0
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        ### charge / discharge should be close to each other to avoid charged storage
        for ii in range(0,8):
            self.assertAlmostEqual(res.x[ii], 0, 3)
        for ii in range(8,10):
            self.assertAlmostEqual(res.x[ii], -5, 3)
        for ii in range(34,36):
            self.assertAlmostEqual(res.x[ii], 5, 3)
        for ii in range(36,48):
            self.assertAlmostEqual(res.x[ii], 0, 3)                   
        # total value: earnung 10+1e5, costs for storage .5 per MWh in storage     
        self.assertAlmostEqual(res.value, 1e4-(5+10+5)*.5-10*1-10*2, 3)

        WACC = 0.8
        ### (B) WITH discount
        a2 = eao.assets.Storage('STORAGE', node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10,\
             cap_in=5, cap_out=5, start_level=0, end_level=0, price='price',
             cost_store= .5, eff_in= 1, cost_in=1, cost_out=2, wacc = WACC)
        op2 = a2.setup_optim_problem(prices, timegrid=timegrid)
        res2 = op2.optimize()
        ### dispatch should not have changed due to discount
        for ii in range(0,48):
            self.assertAlmostEqual(res.x[ii]-res2.x[ii], 0., 3)        
        # total value: earnung 10+1e5, costs for storage .5 per MWh in storage    
        timegrid.set_wacc(WACC) 
        disc = timegrid.discount_factors
        fill_level = -(res2.x[0:24]+res2.x[24:48]).cumsum()
        costs = -res2.x[0:24]*a.cost_in \
                + res2.x[24:48]*a.cost_out \
                + fill_level*a.cost_store
        costs = (costs*disc).sum()
        rev = ((res2.x[24:48])*1e3*disc).sum()
        
        self.assertAlmostEqual(res2.value, rev-costs, 1)        
        pass

class TestOrderOrderBooks(unittest.TestCase):
    def test_order(self):
        """ Unit test. Basic functioning
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        tg2 = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'h')

        ## define order book
        ob = dict( start = [pd.Timestamp(2021,1,1),
                            pd.Timestamp(2021,1,2),
                            pd.Timestamp(2021,1,3),
                            pd.Timestamp(2021,1,7)],
                   end   = [pd.Timestamp(2021,1,2),
                            pd.Timestamp(2021,1,5),
                            pd.Timestamp(2021,1,5),
                            pd.Timestamp(2021,1,15)],
                    capa = [-1.1,
                            -2.2,
                            3.3,
                            4.4],
                    price =[ 8,
                            12,
                            13,
                             9])


        a = eao.assets.SimpleContract(name = 'SC', 
                                      nodes = node,
                                      price = 'market',
                                      min_cap= -100., max_cap=100.)
        order_book = eao.assets.OrderBook(orders=ob, nodes = node)
        portf = eao.portfolio.Portfolio([a, order_book])

        # test with orderbook being a dataframe and missing key
        ob_df = pd.DataFrame(ob)
        ob2   = eao.assets.OrderBook(orders=ob_df)
        ob_missing = ob.copy()
        ob_missing.pop('start')
        try: ob3   = eao.assets.OrderBook(orders=ob_missing)
        except: pass # should fail
        else: raise ValueError('Should have raised an arror')
        prices ={'market': 10*np.ones(timegrid.T)}

        c1 = order_book.setup_optim_problem(prices, timegrid=timegrid, costs_only=True)
        c2 = order_book.setup_optim_problem(prices, timegrid=tg2, costs_only=True)
        np.testing.assert_almost_equal(c1, c2, 3)

        # hourly or daily - costs should be the same
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf= portf, op=op, res=res)
        self.assertAlmostEqual(out['dispatch'].sum()[1], 4.4*24*8 - 2.2*3*24 , 3)   
        self.assertAlmostEqual(res.value, 1161.6, 3)   

    def test_order_serialize(self):
        """ Unit test. serialization
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')

        ## define order book
        ob = dict( start = [pd.Timestamp(2021,1,1),
                            pd.Timestamp(2021,1,2),
                            pd.Timestamp(2021,1,3),
                            pd.Timestamp(2021,1,7)],
                   end   = [pd.Timestamp(2021,1,2),
                            pd.Timestamp(2021,1,5),
                            pd.Timestamp(2021,1,5),
                            pd.Timestamp(2021,1,15)],
                   capa  = [-1.1,
                            -2.2,
                            3.3,
                            4.4],
                   price = [ 8,
                            12,
                            13,
                             9])
        a = eao.assets.SimpleContract(name = 'SC', 
                                      nodes = node,
                                      price = 'market',
                                      min_cap= -100., max_cap=100.)
        
        order_book = eao.assets.OrderBook(orders=ob, nodes = node)

        portf = eao.portfolio.Portfolio([a,order_book])
        s = eao.serialization.to_json(order_book)
        ob = eao.serialization.load_from_json(s)
        portf2 = eao.portfolio.Portfolio([a,ob])
        
        prices ={'market': 10*np.ones(timegrid.T)}
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        op2 = portf2.setup_optim_problem(prices, timegrid=timegrid)
        res2 = op2.optimize()
        self.assertAlmostEqual(res.value, res2.value, 3)

    def test_order_enforce_allatonce(self):
        """ Test enforcing full execution of orders """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'h')

        ## define order book
        ob = dict( start = [pd.Timestamp(2021,1,1, 10),
                            pd.Timestamp(2021,1,2),
                            pd.Timestamp(2021,1,3),
                            pd.Timestamp(2021,1,7)],
                   end   = [pd.Timestamp(2021,1,6),
                            pd.Timestamp(2021,1,5, 20),
                            pd.Timestamp(2021,1,5),
                            pd.Timestamp(2021,1,15)],
                    capa = [6,
                            8,
                            6,
                            7],
                    price =[ 1,
                             2,
                             3,
                             2])
        a = eao.assets.SimpleContract(name = 'SC', 
                                      nodes = node,
                                      price = 'market',
                                      min_cap= -10., max_cap=0) # restricted sale
        
        order_book = eao.assets.OrderBook(orders=ob, nodes = node)

        portf = eao.portfolio.Portfolio([a,order_book])
        
        prices ={'market': 10*np.ones(timegrid.T)} # prices above buy prices, so will want to sell
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf= portf, op=op, res=res)
        # check all is dispatched
        self.assertAlmostEqual(out['dispatch'].sum().iloc[1], 2372, 3)
        self.assertAlmostEqual(out['dispatch'].max().iloc[1], 10, 3) # given by market, orders partly executed
        # now enforce ecex all
        order_book = eao.assets.OrderBook(orders=ob, nodes = node, full_exec=True)
        portf = eao.portfolio.Portfolio([a,order_book])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf= portf, op=op, res=res)
        self.assertAlmostEqual(out['dispatch'].sum().iloc[1], 2004, 3)
        self.assertAlmostEqual(out['dispatch'].max().iloc[1], 7, 3) # given by market, orders partly executed

    def test_order_book_battery(self):
        """ Test enforcing full execution of orders """
        node = eao.assets.Node('power')
        S = dt.date(2021,1,1)
        E = dt.date(2021,1,3)
        timegrid = eao.assets.Timegrid(S, E, freq = 'h')
        # create larger number of orders
        ## some time steps, buy & sell
        ob = pd.DataFrame(columns = ['start', 'end', 'capa', 'price']) # alternative to dict is DataFrame ... converted in asset
        r = dict() # row
        ### orders
        # orders with bid/ask spread on base signal
        # base signal
        bs = (20*np.sin(timegrid.I/4)+20).round(0)
        prices = {'av': bs}
        for ii in timegrid.I:
            tp = timegrid.timepoints[ii]
            # SELL
            for i in range(0,2):
                r['start']   = tp
                r['end']     = tp + pd.Timedelta(3, 'h')
                r['capa']    = float(i)
                r['price']   = bs[ii] + float(i)
                ob.loc[len(ob)] = r
            # BUY
            for i in range(0,2):
                r['start']   = tp
                r['end']     = tp + pd.Timedelta(4, 'h')
                r['capa']    = -float(i)
                r['price']   = bs[ii] + float(i) + 5
                ob.loc[len(ob)] = r            
        # battery
        b = eao.assets.Storage('battery', node, cap_in  = 10, 
                                                cap_out = 10,
                                                size    = 40)
        # last resort - battery end level. May allow battery not to be completely full, "borrowing" in last hours
        a = eao.assets.SimpleContract('fill_level_adjust', node,
                                      max_cap=10,
                                      min_cap=0,
                                      start=timegrid.timepoints[-2],
                                      end=E,
                                      price = 'av',
                                      extra_costs = 10
                                      )
        order_book = eao.assets.OrderBook('orders', node, 
                                          orders=ob, 
                                          full_exec = True)  # not exactly required to relax problem
        portf = eao.portfolio.Portfolio([b, order_book,a])
        op = portf.setup_optim_problem(prices=prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf= portf, op=op, res=res)
        self.assertAlmostEqual(out['dispatch'].abs().sum().sum(), 192, 3) # recursion test


###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
