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

class BatteryTest(unittest.TestCase):
    def test_optimization(self):
        """ trivial test with eff_out
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
        res = op.optimize()
        xin = res.x[0:24]
        xout = res.x[24:48]
        fl = a.fill_level(op, res)
        self.assertAlmostEqual(-xin.sum()/xout.sum(), 1/.9/.8, 3) # overall loss
        self.assertAlmostEqual(fl.max(), 5, 5)
        print(res)


    def test_no_cycles(self):
        """ trivial test max number cycles
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,10), freq = 'h')
        a = eao.assets.Storage('STORAGE', node, 
                               size=2,
                               cap_in=1,
                               cap_out=1, 
                               start_level=0, 
                               end_level=0, 
                               price='price',
                               max_cycles_no = 1.1,
                               max_cycles_freq = 'd')
        price = np.sin(np.linspace(0,200, timegrid.T))+3
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        n = int(timegrid.T)
        xin = res.x[0:n]
        xout = res.x[n:2*n]
        fl = a.fill_level(op, res)
        myrange = pd.date_range(start = timegrid.start, 
                                end =   timegrid.end + pd.Timedelta('1d'), 
                                freq =  'd', 
                                inclusive = 'both')
        for i in range(0,len(myrange)-1):
            myI = (timegrid.timepoints >= myrange[i]) & (timegrid.timepoints < myrange[i+1])
            if any(myI):
                print(abs(xin[myI].sum()))
                self.assertTrue(abs(xin[myI].sum()) <= a.max_cycles_no*a.size/0.9 + 0.0001)


    def test_two_versions(self):
        """ implementing two alternative ways - new battery and via contract max_take / reformulation of roundtrip efficiency
        """
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,10), freq = '2h')
        np.random.seed(2709)
        buy  = np.random.randn(timegrid.T).cumsum()
        sell = buy - 0.2*np.random.rand(timegrid.T)
        prices ={ 'buy': buy, 'sell':sell}

        ######################### settings
        battery_data = {} # capacity, size and efficiency of an on-site battery
        battery_data['cap']  =  1 # MW
        battery_data['size'] = 2 * battery_data['cap'] # 2 hours
        battery_data['eff_in']  = .8
        battery_data['eff_out']  = .9
        battery_data['max_roundtrip'] = 2.2
        battery_data['max_roundtrip_freq'] = 'd'
        battery_data['simult_in_out'] = True
        ### Structural setup, distinguishing own assets and supply from the grid
        node_power    = eao.assets.Node('behind meter')

        myrange = pd.date_range(start = timegrid.start, 
                                end =   timegrid.end + pd.Timedelta('10d'), 
                                freq =  battery_data['max_roundtrip_freq'], 
                                inclusive = 'both')
        max_take= eao.StartEndValueDict(start=myrange.values[0:-1],
                                        end=myrange.values[1:], 
                                        values=np.ones(len(myrange)-1)*battery_data['max_roundtrip']*battery_data['size']/battery_data['eff_in'])

        buy  = eao.assets.SimpleContract(name       = 'buy', 
                                         nodes      = node_power,
                                         price      = 'buy',
                                         min_cap    = 0,
                                         max_cap    =  1000)
        buy_max_take  = eao.assets.Contract(name       = 'buy', 
                                         nodes      = node_power,
                                         price      = 'buy',
                                         min_cap    = 0,
                                         max_cap    =  1000,
                                         max_take   = max_take)                                         
        sell = eao.assets.SimpleContract(name       = 'sell', 
                                         nodes      = node_power,
                                         price      = 'sell',
                                         min_cap    = -1000,
                                         max_cap    =  0)                                         
        ### Our battery
        battery     = eao.assets.Storage(       name       = 'battery',
                                                nodes      = node_power,
                                                cap_in     = battery_data['cap'],
                                                cap_out    = battery_data['cap'],                                        
                                                eff_in     = battery_data['eff_in']*battery_data['eff_out'],
                                                size       = battery_data['size']*battery_data['eff_out'],
                                                start_level= 0.5 * battery_data['size']*battery_data['eff_out'],
                                                end_level  = 0.5 * battery_data['size']*battery_data['eff_out'],
                                                no_simult_in_out = battery_data['simult_in_out'], 
                                                block_size = '2d') 

        battery_new = eao.assets.Storage(       name       = 'battery',
                                                nodes      = node_power,
                                                cap_in     = battery_data['cap'],
                                                cap_out    = battery_data['cap'],                                        
                                                eff_in     = battery_data['eff_in'],
                                                eff_out    = battery_data['eff_out'],
                                                size       = battery_data['size'],
                                                start_level= 0.5 * battery_data['size'],
                                                end_level  = 0.5 * battery_data['size'],
                                                max_cycles_no    = battery_data['max_roundtrip'],
                                                max_cycles_freq  = battery_data['max_roundtrip_freq'],
                                                no_simult_in_out = battery_data['simult_in_out'], 
                                                block_size = '2d') 
        portf = eao.portfolio.Portfolio([battery, buy_max_take, sell])        
        portf_new = eao.portfolio.Portfolio([battery_new, buy, sell])        


        out = eao.optimize(portf = portf, timegrid = timegrid, data = prices, solver = 'SCIP')
        new = eao.optimize(portf = portf_new, timegrid = timegrid, data = prices, solver = 'SCIP')
        self.assertAlmostEqual(out['summary'].loc['value','Values'], new['summary'].loc['value','Values'], 4)

        myrange = pd.date_range(start = timegrid.start, 
                                end =   timegrid.end + pd.Timedelta('1d'), 
                                freq =  battery_data['max_roundtrip_freq'], 
                                inclusive = 'both')        
        mymax = battery_data['size']*battery_data['max_roundtrip']/battery_data['eff_in']        
        for i in range(0,len(myrange)-1):
            myI = (timegrid.timepoints >= myrange[i]) & (timegrid.timepoints < myrange[i+1])
            if any(myI):
                mysum = out['internal_variables'].loc[myI,'battery_charge'].sum()
                self.assertGreater(mymax+1e-3, mysum)
                mysum = new['internal_variables'].loc[myI,'battery_charge'].sum()
                self.assertGreater(mymax+1e-3, mysum)

        out['internal_variables'].loc[myI,'battery_charge'].max()
        self.assertGreater(battery_data['size']+1e-3, out['internal_variables'].loc[:,'battery_charge'].max())


    def test_blocks_split(self):
        """ trivial test with eff_out
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,5), freq = '4h')
        a = eao.assets.Storage('STORAGE', node, 
                               size=5,
                               cap_in=1,
                               cap_out=1, 
                               start_level=.5, 
                               end_level=2, 
                               price='price',
                               eff_in=.8, 
                               eff_out=.9,
                               no_simult_in_out = False,
                               block_size='d')
        price = np.ones([timegrid.T])
        price[0:-1:2] = 50

        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        xin = res.x[0:timegrid.T]
        xout = res.x[timegrid.T:]
        fl = a.fill_level(op, res)
        # ensure start and end level
        self.assertAlmostEqual(fl[0]+xin[0]*0.8+xout[0]/0.9, .5, 5)
        self.assertAlmostEqual(fl[-1], 2, 5)
        # ensure end level between blocks
        myi = timegrid.timepoints.hour ==20
        for i in timegrid.I[myi]:
            self.assertAlmostEqual(fl[i], 2, 5)

###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
