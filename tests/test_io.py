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
from eaopack.serialization import json_serialize_objects, json_deserialize_objects
from eaopack.serialization import to_json, load_from_json, run_from_json
from eaopack.network_graphs import create_graph

class IOTests(unittest.TestCase):

    def test_serialize_assets(self):
        """ Unit test. Setting up a simple contract with random prices 
            and check that it buys full load at negative prices and opposite
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.SimpleContract(name = 'SC', price = 'rand_price', nodes = node ,
                        min_cap= -10., max_cap=10., extra_costs=10, start = dt.date(2021,2,1))
        a.set_timegrid(timegrid)
        # check direkt serialzation
        mys = json.dumps(a, indent=4, default=json_serialize_objects)
        aa = json.loads(mys, object_hook=json_deserialize_objects)

        d1 = a.__dict__.copy()
        d1.pop('timegrid', None)
        d1.pop('nodes', None)

        d2 = aa.__dict__.copy()
        d2.pop('timegrid', None)
        d2.pop('nodes', None)

        # basic properties
        check = (d1 == d2)

        #check nodes
        d1 = a.nodes[0].__dict__.copy()
        d2 = aa.nodes[0].__dict__.copy()
        d1.pop('unit', None)
        d2.pop('unit', None)
        check = check and (d1 == d2)

        #check unit
        d1 = a.nodes[0].unit.__dict__.copy()
        d2 = aa.nodes[0].unit.__dict__.copy()
        check = check and (d1 == d2)

        u = eao.assets.Unit()
        mys = json.dumps(u, indent=4, default=json_serialize_objects)
        uu  = json.loads(mys, object_hook=json_deserialize_objects)
        check = check and (u.__dict__ == uu.__dict__)

        n = eao.assets.Node(name = 'a', commodity='b', unit = u)
        mys = json.dumps(n, indent=4, default=json_serialize_objects)
        nn  = json.loads(mys, object_hook=json_deserialize_objects)
        check = check and (n.commodity == nn.commodity) and (n.name == nn.name) \
            and (n.unit.factor == nn.unit.factor) and (n.unit.flow == nn.unit.flow) and (n.unit.volume == nn.unit.volume)

        assert check
        return check

    def test_serialize_portfolio(self):
        """ Unit test serialize portfolio
        """

        node1 = eao.assets.Node('node_1')
        node2 = eao.assets.Node('node_2')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a1 = eao.assets.SimpleContract(name = 'SC_1', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -20., max_cap=20., start = dt.date(2021,1,10), end = dt.date(2021,1,20))
        #a1.set_timegrid(timegrid)
        a2 = eao.assets.SimpleContract(name = 'SC_2', price = 'rand_price_1', nodes = node1 ,
                        min_cap= -5., max_cap=10., extra_costs= 1.)
        #a2.set_timegrid(timegrid)
        a3 = eao.assets.SimpleContract(name = 'SC_3', price = 'rand_price_2', nodes = node2 ,
                        min_cap= -10., max_cap=10., extra_costs= 1., start = dt.date(2021,1,10), end = dt.date(2021,1,25))
        a4 = eao.assets.Transport(name = 'Tr', costs_const= 1., nodes = [node1, node2],
                        min_cap= 0., max_cap=1.)
        a5 = eao.assets.Storage('storage', nodes = node1, \
             start=dt.date(2021,1,1), end=dt.date(2021,2,1),size=10, \
             cap_in=1.0/24.0, cap_out=1.0/24.0, start_level=5, end_level=5)

        #a3.set_timegrid(timegrid)
        prices ={'rand_price_1': np.ones(timegrid.T)*1.,
                'rand_price_2': np.ones(timegrid.T)*10.,
                'rand_price_3': np.random.randn(timegrid.T)*10.
                }
        
        portf = eao.portfolio.Portfolio([a1, a2, a3, a4, a5])
        portf.set_timegrid(timegrid) 

        mys = json.dumps(portf, indent=4, default=eao.serialization.json_serialize_objects)
        aa = json.loads(mys, object_hook=eao.serialization.json_deserialize_objects)
        
        # file activity
        myf = 'test_portf.json'

        to_json(portf,myf)
        x = load_from_json(file_name = myf)
        xx = load_from_json(mys)

        run_from_json(file_name_in = myf, prices = prices, file_name_out = 'test_results.xlsx')
        run_from_json(file_name_in = myf, prices = prices, file_name_out = 'test_results.csv',\
                          format_out='csv', csv_ger= True)
        res = run_from_json(file_name_in = myf, prices = prices)

        # array
        mys = json.dumps(prices, indent=4, default=json_serialize_objects)
        pp  = json.loads(mys, object_hook=json_deserialize_objects)
        check = True
        for pr in prices:
            check = check and all(prices[pr] == pp[pr])

        return check

    def test_create_network(self):
        """ simple test to create network graph """
        myf = 'tests/demo_portf.json'
        portf = load_from_json(file_name= myf)
        create_graph(portf = portf, file_name='tests/test_graph.pdf')
        return True

    def test_create_network_output_no_image(self):
        """ simple test to create network graph """
        myf = 'tests/demo_portf.json'
        portf = load_from_json(file_name= myf)
        res = create_graph(portf = portf, no_image_output=True)
        assert res['nodes'][0]['id'] == 'location A'
        return True

    def test_dates_serialize(self):
        """ test how to ensure correct serialization of dates in arrays """

        ### capture asset and write to JSON

        ################################### define parameters
        asset_file = os.path.join(os.path.join(os.path.dirname(__file__)),'test_result_asset.JSON')
        node_main     = eao.assets.Node(name = 'main')
        node_internal = eao.assets.Node(name = 'int')
        battery = eao.assets.Storage(name        = 'battery',
                                    nodes       = node_internal,
                                    cap_out     = 3.,
                                    cap_in      = 2.15,
                                    size        = 11,
                                    start_level = 2.5,
                                    end_level   = 2.5,
                                    block_size  = '2d')  # 2 daily blocks
        # charging -- with maximum volume transported
        # since I have a daily restriction, I need to provide it for all days. I choose a validity for the asset
        # as "daily" is not implemented in the asset (yet)
        Start = dt.date(2021,1,1)
        End   = dt.date(2021,1,10)
        dates = pd.date_range(Start, End, freq = 'd').values
        maxCharge = {'start'  : dates[:-1],
                    'end'    : dates[1:],
                    'values' : [3]*(len(dates)-1)}
        charge = eao.assets.ExtendedTransport(name     = 'charge',
                                            min_cap  = 0.,
                                            max_cap  = 12.,
                                            nodes    = [node_main, node_internal],
                                            max_take = maxCharge)  
        # discharging -- no restriction
        discharge = eao.assets.ExtendedTransport(name     = 'discharge',
                                                min_cap  = 0.,
                                                max_cap  = 23.,
                                                nodes    = [node_internal, node_main])

        market = eao.assets.SimpleContract(max_cap=10, min_cap=-10, price='price', nodes = node_main, name = 'market')
        portf          = eao.portfolio.Portfolio([battery, charge, discharge])
        struct_battery = eao.assets.StructuredAsset(name = 'xxxxx', portfolio= portf, nodes = node_main)
        ## write to JSON
        eao.serialization.to_json(struct_battery, file_name=asset_file)
        ## get from JSON
        myasset = eao.serialization.load_from_json( file_name=asset_file)

        # check that dates remain the same
        dates1 = struct_battery.portfolio.assets[1].max_take['start']
        dates2 = myasset.portfolio.assets[1].max_take['start']
        assert((dates1==dates2).all())
        portf = eao.portfolio.Portfolio([market, myasset])

        ## now optimize
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,10), freq = 'd')
        prices ={'price': -(5+5*(np.cos(np.linspace(0.,10., timegrid.T)))) }
        
        optprob = portf.setup_optim_problem(prices = prices, timegrid=timegrid)
        res = optprob.optimize()
        out = eao.io.extract_output(portf = portf, res = res, op = optprob)
        ### assert every second day charging status is 2.5
        fill_level = -out['dispatch']['xxxxx'].cumsum()+2.5
        self.assertAlmostEqual(fill_level['2021-01-02'], 2.5, 5)
        self.assertAlmostEqual(fill_level['2021-01-04'], 2.5, 5)
        self.assertAlmostEqual(fill_level['2021-01-06'], 2.5, 5)
        self.assertAlmostEqual(fill_level['2021-01-08'], 2.5, 5)
        self.assertAlmostEqual(fill_level['2021-01-09'], 2.5, 5)        

class OptimizeShortcutTests(unittest.TestCase):
    
    def test_optimize_shortcut(self):
        """ Test short cut to call optimization in one line
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
        # op    = portf.setup_optim_problem(prices, timegrid)
        # res = op.optimize()
        out = eao.optimize(portf=portf, data=prices, timegrid=timegrid)
        out = eao.optimize(portf=portf, data=prices, timegrid=timegrid, split_interval_size='d')
        out = eao.optimize(portf=portf, data=prices, timegrid=timegrid, split_interval_size='d', make_soft_problem=True)
        return
    
    def test_not_enough_data(self):
        """ Unit test: cast data into timegrid - check error catching etc """

        tg_data = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,4,1), freq = '15min')
        # random data
        data = pd.DataFrame(
            np.random.randn(len(tg_data.timepoints), 3),
            index=tg_data.timepoints,
            columns=['a', 'b', 'c'])
        # other timegrids ...
        # (1) ssame tg
        tg = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,4,1), freq = '15min')
        df = tg.prices_to_grid(data)
        self.assertTrue((data == df).values.all())
        # (2) smaller tg
        tg = eao.assets.Timegrid(dt.date(2021,2,1), dt.date(2021,3,1), freq = '15min')
        df = tg.prices_to_grid(data)
        ddata = data.loc['2021-02-01 00:00':'2021-02-28 23:45', :]
        self.assertTrue((ddata == df).values.all())
        # (3a) larger tg - throw error! - start
        tg = eao.assets.Timegrid(dt.date(2021,2,1), dt.date(2021,5,1), freq = '15min')
        with self.assertRaises(Exception):
            df = tg.prices_to_grid(data)
        # (3b) larger tg - throw error! - end
        tg = eao.assets.Timegrid(dt.date(2020,12,31), dt.date(2021,3,1), freq = '15min')
        with self.assertRaises(Exception):
            df = tg.prices_to_grid(data)            


if __name__ == "__main__" :
    unittest.main()

