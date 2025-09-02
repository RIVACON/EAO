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

class MIP(unittest.TestCase):
    """ Tests MIP """
    def test_hello_MIP(self):
        """ artificial MIP
        """
        node = eao.assets.Node('N1')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.SimpleContract(name = 'c', max_cap=1.1, price='price', nodes = node)        
        price = np.ones([timegrid.T])
        price[:10] = -1
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        ### create booleans in mapping
        op.mapping['bool'] = False
        op.mapping.loc[0:4,'bool'] = True
        op.u[0:5] = 1.6  # Account for different optim interfaces that would treat this either as a bool or as an int variable
        res = op.optimize()
        self.assertAlmostEqual(res.value, 5*1 + 1.1*24*5, 5) 

    def test_MIP_storage(self):
        """ MIP storage
        """
        ### (1) without MIP constraint --- will "burn" electricity by simultaneous charging/discharging
        node = eao.assets.Node('N1')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,10), freq = 'd')
        a = eao.assets.Storage(name = 'c', cap_in=1, cap_out=1, size = 1, price='price', nodes = node, eff_in= 0.5, no_simult_in_out= False)        
        price = np.ones([timegrid.T])
        price[:3] = -1
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res1 = op.optimize()
        # hold against trivial solution: (burn electricity where prices are negative - and at end use battery volume)
        self.assertAlmostEqual(res1.value, 38., 5) 
        sol = np.asarray([[-24., -24., -24.,   0.,   0.,   0.,   0.,   0.,   0.],
                          [ 12.,  12.,  11.,   1.,   0.,   0.,   0.,   0.,   0.]])
        check = res1.x.reshape((2,9))
        ### attention - freedom of choice, not exact match needed
        self.assertAlmostEqual(check[0,0:3].sum(), sol[0,0:3].sum() , 5) 
        self.assertAlmostEqual(check[1,0:3].sum(), sol[1,0:3].sum() , 5)         
        self.assertAlmostEqual(check[0,3:9].sum(), sol[0,3:9].sum() , 5)
        self.assertAlmostEqual(check[1,3:9].sum(), sol[1,3:9].sum() , 5)        
        ### (2) with  MIP constraint --- will NOT "burn" electricity by simultaneous charging/discharging
        a = eao.assets.Storage(name = 'c', cap_in=1, cap_out=1, size = 1, price='price', nodes = node, eff_in= 0.5, no_simult_in_out= True)        
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res2 = op.optimize()
        # hold against trivial solution: 
        self.assertAlmostEqual(res2.value, 4, 5) 
        sol = np.asarray([[-2.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  0.],
                          [ 0.,  1., -0., -0., -0., -0., -0., -0.,  1.],
                          [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
        check = res2.x.reshape((3,9))
        # charge - discharge - charge must be given in 0:3
        for i in range(0,3):
            for j in range(0,3):
                self.assertAlmostEqual(check[i,j], sol[i,j] , 5) 
        ### discharge any time >= step 3
        self.assertAlmostEqual(check[0,3:9].sum(), sol[0,3:9].sum() , 5)
        self.assertAlmostEqual(check[1,3:9].sum(), sol[1,3:9].sum() , 5)        

    def test_MIP_storage(self):
        """  with portfolio and MIP constraint --- but positive prices --> both solutions should be identical """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,2), freq = 'h')
        prices ={ 'price': 40-20*np.cos(np.linspace(0,20, timegrid.T))}
        # (1) no MIP
        a = eao.assets.Storage('STORAGE', nodes = node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10,\
             cap_in=5, cap_out=5, start_level=0, end_level=0, 
             cost_store= .1, eff_in= .8, cost_in=1, cost_out=1)
        m = eao.assets.SimpleContract(name = 'market', nodes = node, min_cap = -11, max_cap = 12, price='price')
        portf1 = eao.portfolio.Portfolio([a,m])
        op1 = portf1.setup_optim_problem(prices, timegrid=timegrid)
        res1 = op1.optimize()
        # (1) same as MIP
        a2 = eao.assets.Storage('STORAGE', nodes = node, start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10,\
             cap_in=5, cap_out=5, start_level=0, end_level=0, 
             cost_store= .1, eff_in= .8, cost_in=1, cost_out=1, 
             no_simult_in_out= True)
        m = eao.assets.SimpleContract(name = 'market', nodes = node, min_cap = -11, max_cap = 12, price='price')
        portf2 = eao.portfolio.Portfolio([a2,m])
        op2 = portf2.setup_optim_problem(prices, timegrid=timegrid)
        res2 = op2.optimize()
        ## assert 1 == 2
        self.assertAlmostEqual(res1.value, res2.value , 5)
        ### check dispatch
        out1 = eao.io.extract_output(portf = portf1, op = op1, res = res1)
        out2 = eao.io.extract_output(portf = portf2, op = op2, res = res2)
        out2 = eao.io.extract_output(portf = portf2, op = op2, res = res2)
        self.assertAlmostEqual((out1['dispatch']-out2['dispatch']).sum().sum(), 0, 5)
        ### functionality checks
        eao.io.output_to_file(out2, file_name= 'test_results.xlsx')
        test_string = eao.serialization.json_serialize_objects(portf2)
        # check variable naming (for RI etc)
        map = op2.mapping[op2.mapping['asset']=='STORAGE']
        assert (len(map) == 72), 'wrong length of mapping (2x disp plus bool)'
        assert all(map['var_name'].values[0:24]=='disp_in')
        assert all(map['var_name'].values[24:48]=='disp_out')
        assert all(map['var_name'].values[48:72]=='bool_1')

    def test_MIP_storage_storage_duration(self):
        """ MIP storage. addtl feature - limited storage duration
        """
        node = eao.assets.Node('N1')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,2), freq = 'h')
        # (1) case one var per time step
        a = eao.assets.Storage(name = 'c', cap_in=1, cap_out=1, size = 10, price='price', nodes = node, 
                               no_simult_in_out= False, max_store_duration= 3)        
        price = np.ones([timegrid.T])
        price[:10] = -1
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        check = res.x.reshape((2,24)).T
        fill_level = -check[:,0].cumsum()
        for (f,b) in zip(fill_level, check[:,1]):
            if (f>1e-8) and (b==0):
                raise ValueError('fill level non zero, but bool zero')
        for ii in range(0,(24-4)):
            assert (check[ii:ii+4,1]).sum()<=3
        # (2) case two var per time step (with efficiency)
        a = eao.assets.Storage(name = 'c', cap_in=1, cap_out=1, size = 10, price='price', nodes = node, eff_in= 0.8,
                               no_simult_in_out= False, max_store_duration= 3)        
        price = np.ones([timegrid.T])
        price[:10] = -1
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        check = res.x.reshape((3,24)).T
        fill_level = -(check[:,0]*0.8+check[:,1]).cumsum()
        for (f,b) in zip(fill_level, check[:,2]):
            if (f>1e-5) and (b==0):
                raise ValueError('fill level non zero, but bool zero')
        for ii in range(0,(24-4)):
            assert (check[ii:ii+4,2]).sum()<=3        
        # (3) case with ensure no ...
        a = eao.assets.Storage(name = 'c', cap_in=1, cap_out=1, size = 10, price='price', nodes = node, eff_in= 0.8,
                               no_simult_in_out= True, max_store_duration= 3)        
        price = np.ones([timegrid.T])
        price[:10] = -1
        prices ={ 'price': price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        check = res.x.reshape((4,24)).T
        fill_level = -(check[:,0]*0.8+check[:,1]).cumsum()
        for (f,b) in zip(fill_level, check[:,3]):
            if (f>1e-5) and (b==0):
                raise ValueError('fill level non zero, but bool zero')
        for ii in range(0,(24-4)):
            assert (check[ii:ii+4,3]).sum()<=3 + 1e-4

    def test_max_hold_in_portfolio(self):
        node1 = eao.assets.Node('N1')
        node2 = eao.assets.Node('N2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,5), freq = '3h')
        a = eao.assets.Storage('a1', [node1, node2], start=dt.datetime(2020,12,31,23), end=dt.date(2021,2,1),size=10,\
             cap_in=1, cap_out=1, start_level=0, end_level=0, cost_in=.1, eff_in= 0.7,
             no_simult_in_out=True, max_store_duration= 7)
        buy  = eao.assets.SimpleContract(name = 'buy', nodes = node1, min_cap=-10, max_cap=10, price = 'price')
        sell = eao.assets.SimpleContract(name = 'sell', nodes = node2, min_cap=-10, max_cap=10, price = 'price')        
        # cannot be used, since storage can only BRING volumes from node1 to node2
        portf = eao.portfolio.Portfolio([a, buy, sell])
        test_string = eao.serialization.json_serialize_objects(portf)
        price = -np.sin(np.linspace(0,12,timegrid.T))
        prices ={ 'price': price, 'zero':np.zeros(timegrid.T), 'best':-np.ones(timegrid.T)}
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf = portf, op = op, res = res, prices = prices)
        out['dispatch']['fill_level'] = -(out['dispatch']['a1 (N2)']+out['dispatch']['a1 (N1)']*0.7).cumsum()
        out['dispatch'] = out['dispatch'].round(5)
        # basic checks
        self.assertAlmostEqual(out['dispatch']['fill_level'].iloc[0],0,5)
        assert out['dispatch']['fill_level'].max()<=10
        assert out['dispatch']['fill_level'].min()>=0
        # no simultaneaous
        assert not any((out['dispatch']['a1 (N1)'] !=0) & (out['dispatch']['a1 (N2)'] !=0))
        # duration fill level non zero smaller 7 hours
        dur = 0
        for i,r in out['dispatch'].iterrows():
            if r.fill_level>0:
                dur +=3
            else:
                dur = 0
            assert dur <=7
                 
###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
