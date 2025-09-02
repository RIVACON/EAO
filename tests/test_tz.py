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

import pytz

class TimeZones(unittest.TestCase):
    def test_opt_tz(self):
        """ Unit test. Setting up a simple contract and run with timezone. No change outside change summer/winter time
        """
        node1a = eao.assets.Node('N1a')
        node1b = eao.assets.Node('N1b')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,11), freq = 'h', main_time_unit='h', timezone = 'CET')
        prices ={'buy': np.ones(timegrid.T), 'sell': 10.*np.ones(timegrid.T)}
        trans = eao.assets.Transport(name = 'TrA', nodes = [node1a, node2],
                        min_cap= 0., max_cap=10., efficiency = 0.95)
        buy  = eao.assets.SimpleContract(name = 'buya', price = 'buy', max_cap = 1, nodes = node1a, start = dt.date(2021,1,2)  )
        sell = eao.assets.SimpleContract(name = 'sell', price = 'sell', min_cap = -1, nodes = node2  )

        portf = eao.portfolio.Portfolio([trans, buy, sell])
        tgCET  = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,11), freq = 'h', main_time_unit='h', timezone = 'CET')
        tgNone  = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,11), freq = 'h', main_time_unit='h', timezone = 'CET')        
        opCET = portf.setup_optim_problem(prices, tgCET)
        resCET = opCET.optimize()
        opNone = portf.setup_optim_problem(prices, tgNone)
        resNone = opNone.optimize()

        self.assertAlmostEqual(resCET.value, resNone.value, 5) # no change in summer/winter time --> same result

    def test_contract_with_max_take(self):
        """ Unit test. Extended transport
        """
        node = eao.assets.Node('N1a')
        Start = dt.datetime(2021,3,27) # includes time change winter -> summer
        End   = dt.date(2021,3,29)

        # I expect so many hours between start and end
        h_naive = 48
        h_cet   = 47

        tgNaive = eao.assets.Timegrid(Start, End , freq = 'h', main_time_unit='h', timezone= None)
        self.assertEqual(tgNaive.T, h_naive)
        tgCET = eao.assets.Timegrid(Start, End , freq = 'h', main_time_unit='h', timezone= 'CET')
        self.assertEqual(tgCET.T, h_cet)

        min_take = {}
        min_take['start']  = Start
        min_take['end']    = End
        min_take['values'] = -10.

        prices ={'buy': np.ones(tgCET.T), 'sell': 10.*np.ones(tgCET.T)}
        buy  = eao.assets.SimpleContract(name = 'buya', price = 'buy', max_cap = 1, nodes = node)
        sell = eao.assets.Contract(name = 'sell', price = 'sell', min_cap = -1, min_take = min_take, nodes = node)

        portf = eao.portfolio.Portfolio([buy, sell])
        op = portf.setup_optim_problem(prices, tgCET)
        res = op.optimize()

        self.assertAlmostEqual(res.value, 90., 5) # given by min_take
        out = eao.io.extract_output(portf, op, res, prices)
        eao.io.output_to_file(out, 'test_output.xlsx')

    def test_tz_serialization(self):
        """ Unit test. Timezones and serialization
        """
        node = eao.assets.Node('N1a')
        Start = dt.datetime(2020,10,24) # includes time change winter -> summer
        #timezone = pytz.timezone("CET")
        #Start = timezone.localize(Start)
        End   = dt.date(2020,10,26)

        tgCET = eao.assets.Timegrid(Start, End , freq = 'h', main_time_unit='h', timezone= 'CET')

        # simple check - number of hours to expect
        h_cet   = 49
        self.assertEqual(tgCET.T, h_cet)

        min_take = {}
        min_take['start']  = Start
        min_take['end']    = End
        min_take['values'] = -10.

        prices ={'buy': np.ones(tgCET.T), 'sell': 10.*np.ones(tgCET.T)}
        buy  = eao.assets.SimpleContract(name = 'buya', price = 'buy', max_cap = 1, nodes = node)
        sell = eao.assets.Contract(name = 'sell', price = 'sell', min_cap = -1, min_take = min_take, nodes = node)

        portf = eao.portfolio.Portfolio([buy, sell])
        op = portf.setup_optim_problem(prices, tgCET)
        res = op.optimize()

        # serialize / deserialize
        ss = eao.serialization.to_json(portf)
        portf_io = eao.serialization.load_from_json(ss)
        
        # test with hourly prices -  as list
        myd =  pd.date_range(Start, End, freq = '15min', tz = 'CET').tolist()
        ss = eao.serialization.to_json(myd)
        myd2 = eao.serialization.load_from_json(ss)
        assert myd == myd2
        # as index
        myd =  pd.date_range(Start, End, freq = '15min', tz = 'CET')
        ss = eao.serialization.to_json(myd)
        myd2 = eao.serialization.load_from_json(ss)
        assert all(myd2 == myd)

        ### serialize contract
        min_take = {}
        min_take['start']  = pd.date_range(Start, End, freq = '15min', tz = 'CET')
        min_take['end'] = min_take['start']
        min_take['values'] = -10.*np.ones(len(min_take['start']))
        sell = eao.assets.Contract(name = 'sell', price = 'sell', min_cap = -1, min_take = min_take, nodes = node)
        ss = eao.serialization.to_json(sell)
        c2 = eao.serialization.load_from_json(ss)
        for a,b in zip(c2.min_take['start'], sell.min_take['start']):
            assert(a==b)
        assert (c2.min_take['start'][-1] == pd.Timestamp(End, tz = 'CET'))


    def test_tz_to_grid(self):
        """ Unit test. Extended transport
        """
        node = eao.assets.Node('N1a')

        Start = dt.datetime(2020,10,24) # includes time change winter -> summer
        End   = dt.date(2020,10,26)
        tgCET = eao.assets.Timegrid(Start, End , freq = '10min', main_time_unit='h', timezone= 'CET')
        mybuy = {}
        mybuy['start']  = pd.date_range(Start, End, freq = 'h', tz = 'CET')
        mybuy['values'] = np.random.rand(len(mybuy['start']))
        mybuy_grid = tgCET.values_to_grid(mybuy)
        prices ={'buy': mybuy_grid, 'sell': np.random.rand(tgCET.T)}

        ss = eao.serialization.to_json(prices)
        p2 = eao.serialization.load_from_json(ss)

        assert all(prices['buy']==p2['buy'])
        assert all(prices['sell']==p2['sell'])

        Start = dt.datetime(2020,3,20) # includes time change winter -> summer
        End   = dt.date(2020,4,1)
        tgCET = eao.assets.Timegrid(Start, End , freq = '10min', main_time_unit='h', timezone= 'CET')
        mybuy = {}
        mybuy['start']  = pd.date_range(Start, End, freq = 'h', tz = 'CET')
        mybuy['values'] = np.random.rand(len(mybuy['start']))
        mybuy_grid = tgCET.values_to_grid(mybuy)
        prices ={'buy': mybuy_grid, 'sell': np.random.rand(tgCET.T)}

        ss = eao.serialization.to_json(prices)
        p2 = eao.serialization.load_from_json(ss)

        assert all(prices['buy']==p2['buy'])
        assert all(prices['sell']==p2['sell'])


    def test_optim_trivial_tz(self):
        """Simple test where first ten times price is zero and afterwards price is one, zero costs
        """
        ### march
        node = eao.assets.Node('testNode')
        Start = dt.datetime(2020,3,29) # includes time change winter -> summer
        End   = dt.date(2020,3,30)        
        timegrid = eao.assets.Timegrid(Start, End, freq = '15min', timezone='CET')

        a = eao.assets.Storage('STORAGE', node, start=Start, end=End,size=100,\
             cap_in=1, cap_out=1, start_level=0, end_level=0, price='price')
        price = np.ones([timegrid.T])
        price[:46] = 0
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 46./4., 5)
        
        ### october
        node = eao.assets.Node('testNode')
        Start = dt.datetime(2020,10,25) # includes time change winter -> summer
        End   = dt.date(2020,10,26)        
        timegrid = eao.assets.Timegrid(Start, End, freq = '15min', timezone='CET')

        a = eao.assets.Storage('STORAGE', node, start=Start, end=End,size=100,\
             cap_in=1, cap_out=1, start_level=0, end_level=0, price='price')
        price = np.ones([timegrid.T])
        price[:50] = 0
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(res.value, 50./4., 5)

                       

###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
