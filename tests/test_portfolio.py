import numpy as np
import pandas as pd
import datetime as dt
import unittest

import os, sys
mypath = (os.path.dirname(__file__))
sys.path.append(os.path.join(mypath, '..'))

import eaopack as eao




class PortfolioTests(unittest.TestCase):

    
    def test_simple_portfolio(self):
        """ Unit test. Setting up a simple portfolio to check restrictions on nodes and
            other basic functionality
        """

        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=20., start = dt.date(2021,1,10), end = dt.date(2021,1,20))
        #a1.set_timegrid(timegrid)
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_2', nodes = node1 ,
                        min_cap= -5., max_cap=10.)#, extra_costs= 1.)
        #a2.set_timegrid(timegrid)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= -1., max_cap=10., extra_costs= 1.)
        a5 = eao.assets.Storage('storage', nodes = node1, \
             start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
             cap_in=1.0/24.0, cap_out=1.0/24.0, start_level=5, end_level=5)
        #a3.set_timegrid(timegrid)
        prices ={'rand_price_1': np.random.rand(timegrid.T)-0.5,
                'rand_price_2': 5.*(np.random.rand(timegrid.T)-0.5),
                }
        
        portf = eao.portfolio.Portfolio([a1, a2, a3, a5])
        op    = portf.setup_optim_problem(prices, timegrid)
        res = op.optimize()

        check = True # simple run-through test
        return check


    def test_portfolio_with_transport(self):
        """ Unit test. Setting up a simple portfolio to check restrictions on nodes and
            other basic functionality
        """

        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=20., start = dt.date(2021,1,10), end = dt.date(2021,1,20))
        #a1.set_timegrid(timegrid)
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -5., max_cap=10.)#, extra_costs= 1.)
        #a2.set_timegrid(timegrid)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= -10., max_cap=10., extra_costs= 1., start = dt.date(2021,1,10), end = dt.date(2021,1,25))
        a4 = eao.assets.Transport(name = 'Tr', costs_const= 5., nodes = [node1, node2],
                        min_cap= 0., max_cap=1.)

        #a3.set_timegrid(timegrid)
        prices ={'rand_price_1': np.ones(timegrid.T)*1.,
                'rand_price_2': np.ones(timegrid.T)*10.,
                }
        
        portf = eao.portfolio.Portfolio([a1, a2, a3, a4])
        op    = portf.setup_optim_problem(prices, timegrid)
        res = op.optimize()
        ### checks
        # transport constitutes the bottleneck
        x = np.around(res.x, 2)
        mapp = op.mapping
        #### with several rows per variable not valid. Using extraction function
        # disp = pd.DataFrame()
        # II = mapp.index[(mapp.asset == 'Tr')&(mapp.type == 'd')&(mapp.node == 'node_1')]
        # disp['TR n1'] = x[II]*mapp.loc[II, 'disp_factor']
        # disp['TR n2'] = x[mapp.index[(mapp.asset == 'Tr')&(mapp.type == 'd')&(mapp.node == 'node_2')]]
        # check = all(disp['TR n1']+disp['TR n2']==0)  # transport with zero total dispatch in all steps
        out = eao.io.extract_output(portf= portf, op=op, res=res)
        check = (out['dispatch']['Tr (node_1)']+out['dispatch']['Tr (node_2)']).sum()
        self.assertAlmostEqual(check, 0., 5)
        check = (out['dispatch']['Tr (node_2)']+out['dispatch']['SC_3 (node_2)']).sum()
        self.assertAlmostEqual(check, 0., 5)
        check = (out['dispatch']['Tr (node_1)']+out['dispatch']['SC_1 (node_1)']+out['dispatch']['SC_2 (node_1)']).sum()
        self.assertAlmostEqual(check, 0., 5)        
        return

    def test_various_vars_in_mapping(self):
        """ testing more efficient approach where more than one row in mapping
        can exist per variable """
        #unit = eao.assets.Unit(volume='MJ', flow = 'W', factor = 1000) # physically not correct, for testing
        unit = eao.assets.Unit()
        node1 = eao.assets.Node('n1', commodity = 'com', unit = unit)
        node2 = eao.assets.Node('n2', commodity = 'com', unit = unit)

        tg = eao.assets.Timegrid(start = pd.Timestamp(1980, 1, 1), end = pd.Timestamp(1981, 1, 1), freq = 'd', main_time_unit='d')
        b = eao.assets.SimpleContract(name = 'b', nodes = node1, min_cap= -100, max_cap=100, price = 'Z')
        s = eao.assets.SimpleContract(name = 's', nodes = node2, min_cap= -100, max_cap=100, price = 'P')        

        take = {'start': pd.date_range(start = pd.Timestamp(1980, 1, 1), end = pd.Timestamp(1982, 1, 1), freq = 'MS'),
                   'end': pd.date_range(start = pd.Timestamp(1980, 2, 1), end = pd.Timestamp(1982, 2, 1), freq = 'MS'),
                   }
        take['values'] = np.ones(len(take['start']))*22
        t = eao.assets.ExtendedTransport(name = 'trans',  max_cap = 100,  max_take = take, efficiency = 0.9, nodes = [node1, node2])

        price = {'Z': np.zeros(tg.T),
                 'P': np.ones(tg.T)*1}

        portf = eao.portfolio.Portfolio([b, s, t])
        op = portf.setup_optim_problem(timegrid = tg, prices = price)
        res = op.optimize()
        out = eao.io.extract_output(portf = portf, op = op, res = res)
        df = out['dispatch']

        # efficiency right?
        self.assertAlmostEqual(-df['trans (n2)'].sum()/df['trans (n1)'].sum(), 0.9, 5)
        # right monthly quantity?
        df['month'] = df.index.month
        for v in df.groupby('month').sum()['trans (n1)']:
            self.assertAlmostEqual(v, -22, 5)
        

    def test_portf_min_cap_from_price_DF(self):
        """ Unit test. Setting up a simple portfolio and check workflow with restriction from price DF
        """

        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=20.)
        #a1.set_timegrid(timegrid)
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_2', nodes = node1 ,
                        min_cap= 'CAP', max_cap='CAP', start = dt.date(2021,1,10), end = dt.date(2021,1,20), extra_costs=1.)
        #a2.set_timegrid(timegrid)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= -1., max_cap=10., extra_costs= 1.)
        a5 = eao.assets.Storage('storage', nodes = node1, \
             start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
             cap_in=1.0/24.0, cap_out=1.0/24.0, start_level=5, end_level=5)
        #a3.set_timegrid(timegrid)
        prices ={'rand_price_1': np.random.rand(timegrid.T)-0.5,
                'rand_price_2' : 5.*(np.random.rand(timegrid.T)-0.5),
                'CAP'          : np.random.randn(timegrid.T)
                }
        
        portf = eao.portfolio.Portfolio([a1, a2, a3, a5])
        op    = portf.setup_optim_problem(prices, timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        cap_out = out['prices']['input data: CAP'].values
        disp    = out['dispatch']['SC_2 (node_1)'].values/24 # 24h per day!
        # outside start & end zero
        self.assertAlmostEqual(abs(disp[0:9]).sum(), 0., 5)
        self.assertAlmostEqual(abs(disp[19:]).sum(), 0., 5)
        self.assertAlmostEqual(abs(disp[9:19]-prices['CAP'][9:19]).sum(), 0., 5)



class StructuredAssetsTests(unittest.TestCase):
    """ Unit tests for structured assets - defined on basis of portfolios or several assets
    """

    def test_structured_asset(self):
        """ test asset consisting of a portfolio """
    
        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        node3 = eao.assets.Node('node_3')
        import datetime as dt
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
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

        #a3.set_timegrid(timegrid)
        prices ={'rand_price_1': (np.ones(timegrid.T)*1.).tolist(),
                'rand_price_2': (np.ones(timegrid.T)*10.).tolist(),
                }

        portf2 = eao.portfolio.Portfolio([a1, a2, a3, a4,a1a])
        op2 = portf2.setup_optim_problem(prices, timegrid)
        res_separate = op2.optimize()                
        print(res_separate.value)

        portfStr = eao.portfolio.Portfolio([a1, a2,  a4, a1a])
        a = eao.assets.StructuredAsset(name = 'x',portfolio = portfStr, nodes = node2)
        portfStr2 = eao.portfolio.Portfolio([a, a3])
        # same task without wrapping
        opStr = portfStr2.setup_optim_problem(prices, timegrid)
        # test "only costs"
        c_only = portfStr2.setup_optim_problem(prices, timegrid, costs_only = True)
        res_struct = opStr.optimize()
        assert(all(opStr.c == c_only))

        self.assertAlmostEqual(res_struct.value, res_separate.value, 5)        
        outp = eao.io.extract_output(portfStr2, opStr, res_struct)
        eao.io.output_to_file(output=outp, file_name= 'test.xlsx')


    def test_linked_asset(self):
        """ Unit test. Linked asset with asset 1 having higher costs. Runs before asset 2
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'h')
        demand = eao.assets.SimpleContract(name = 'demand', nodes = node_power ,     min_cap= -10., max_cap=-10.)
        a1 =     eao.assets.CHPAsset(name='CHP1', extra_costs = 10, nodes = (node_power, node_heat), min_cap=5., max_cap= 5.)
        a2 =     eao.assets.CHPAsset(name='CHP2', extra_costs = 5, nodes = (node_power, node_heat), min_cap=2., max_cap=15.)
        p = eao.portfolio.Portfolio([a1, a2]) # collect the two assets
        linked = eao.assets.LinkedAsset(p, nodes = [node_power, node_heat], 
                                            asset1_variable=[a2, 'disp', node_power], 
                                            asset2_variable=[a1, 'bool_on', None],
                                            time_back=0,
                                            name = 'extra_CHP')
        check_op = linked.setup_optim_problem(prices = [], timegrid=timegrid)
        portf = eao.portfolio.Portfolio([linked, demand])
        op = portf.setup_optim_problem(prices = [], timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res)
        # if asset a1 must run first, before a2 can gererate, costs must be 5*10 + 5*5 = 75
        self.assertAlmostEqual(np.abs(out['DCF']['extra_CHP'].values + 75.).sum(),0, 4)


###########################################################################################################

###########################################################################################################
###########################################################################################################

if __name__ == "__main__" :
    unittest.main()