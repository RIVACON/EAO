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

class various(unittest.TestCase):
    def test_battery_efficiency(self):
        """ test specific setup of a battery to show the importance of no_simult_in_out.
        """
        s = """{
            "__class__": "Portfolio",
            "assets": [
                {
                    "__class__": "Asset",
                    "asset_type": "Storage",
                    "block_size": null,
                    "cap_in": 50,
                    "cap_out": 50,
                    "cost_in": 0.0,
                    "cost_out": 0.0,
                    "cost_store": 0.0,
                    "eff_in": 0.9,
                    "end": null,
                    "end_level": 50.0,
                    "freq": null,
                    "inflow": 0.0,
                    "max_store_duration": null,
                    "name": "battery",
                    "no_simult_in_out": false,
                    "nodes": [
                        {
                            "__class__": "Node",
                            "commodity": null,
                            "name": "power",
                            "unit": {
                                "__class__": "Unit",
                                "factor": 1.0,
                                "flow": "MW",
                                "volume": "MWh"
                            }
                        }
                    ],
                    "periodicity": null,
                    "periodicity_duration": null,
                    "price": null,
                    "profile": null,
                    "size": 100.0,
                    "start": null,
                    "start_level": 50.0,
                    "wacc": 0.0
                },
                {
                    "__class__": "Asset",
                    "asset_type": "SimpleContract",
                    "end": null,
                    "extra_costs": 0,
                    "freq": null,
                    "max_cap": 500,
                    "min_cap": -500,
                    "name": "supply",
                    "nodes": [
                        {
                            "__class__": "Node",
                            "commodity": null,
                            "name": "power",
                            "unit": {
                                "__class__": "Unit",
                                "factor": 1.0,
                                "flow": "MW",
                                "volume": "MWh"
                            }
                        }
                    ],
                    "periodicity": null,
                    "periodicity_duration": null,
                    "price": "price",
                    "profile": null,
                    "start": null,
                    "wacc": 0
                }
            ]
        }"""
        size = 100 # battery size
        eff = 0.9
        portf = eao.serialization.load_from_json(s)
        portf.assets[0].eff_in = eff
        portf.assets[0].size = size
        portf.assets[0].no_simult_in_out = True # !!! we sum up in and out in output. Computing the storage level won't work without enforcing
        tg = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,3), freq = 'h')
        prices ={ 'price': np.sin(np.linspace(0,40,tg.T))}
        op = portf.setup_optim_problem(timegrid = tg, prices = prices)
        res = op.optimize()
        out = eao.io.extract_output(portf = portf, op = op, res = res, prices=prices)
        # eao.io.output_to_file(out, 'test.xlsx')
        # get fill level from asset
        bat = portf.assets[0]
        fill_level = bat.fill_level(op, res)
        # calculate fill level from dispatch
        fill_level_check = -out['dispatch'].loc[:,'battery']
        fill_level_check[fill_level_check>0] *= eff
        fill_level_check = fill_level_check.cumsum()+50
        self.assertGreaterEqual(np.round(fill_level.min(), 3), 0)
        self.assertGreaterEqual(100, np.round(fill_level.max(), 3))
        self.assertAlmostEqual(abs(fill_level_check.values-fill_level).sum(), 0,4)
        # get fill level from output and check
        fl_out = out['internal_variables'].loc[:,'battery_fill_level'].values
        self.assertAlmostEqual(abs(fill_level-fl_out).sum(), 0,4)
        self.assertAlmostEqual(res.value, 1137.69104689219,3) # reversion test
        self.assertAlmostEqual(out['dispatch'].loc[:,'battery'].sum(), -71.66666667,3) # reversion test
        self.assertAlmostEqual(fill_level[-1], bat.end_level,3)

    def test_battery_efficiency_100(self):
        """ test specific setup of a battery to show the importance of no_simult_in_out.
        """
        s = """{
            "__class__": "Portfolio",
            "assets": [
                {
                    "__class__": "Asset",
                    "asset_type": "Storage",
                    "block_size": null,
                    "cap_in": 50,
                    "cap_out": 50,
                    "cost_in": 0.0,
                    "cost_out": 0.0,
                    "cost_store": 0.0,
                    "eff_in": 0.9,
                    "end": null,
                    "end_level": 50.0,
                    "freq": null,
                    "inflow": 0.0,
                    "max_store_duration": null,
                    "name": "battery",
                    "no_simult_in_out": false,
                    "nodes": [
                        {
                            "__class__": "Node",
                            "commodity": null,
                            "name": "power",
                            "unit": {
                                "__class__": "Unit",
                                "factor": 1.0,
                                "flow": "MW",
                                "volume": "MWh"
                            }
                        }
                    ],
                    "periodicity": null,
                    "periodicity_duration": null,
                    "price": null,
                    "profile": null,
                    "size": 100.0,
                    "start": null,
                    "start_level": 50.0,
                    "wacc": 0.0
                },
                {
                    "__class__": "Asset",
                    "asset_type": "SimpleContract",
                    "end": null,
                    "extra_costs": 0,
                    "freq": null,
                    "max_cap": 500,
                    "min_cap": -500,
                    "name": "supply",
                    "nodes": [
                        {
                            "__class__": "Node",
                            "commodity": null,
                            "name": "power",
                            "unit": {
                                "__class__": "Unit",
                                "factor": 1.0,
                                "flow": "MW",
                                "volume": "MWh"
                            }
                        }
                    ],
                    "periodicity": null,
                    "periodicity_duration": null,
                    "price": "price",
                    "profile": null,
                    "start": null,
                    "wacc": 0
                }
            ]
        }"""
        size = 100 # battery size
        eff = 1
        portf = eao.serialization.load_from_json(s)
        portf.assets[0].eff_in = eff
        portf.assets[0].size = size
        portf.assets[0].no_simult_in_out = False
        tg = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,3), freq = 'h')
        prices ={ 'price': np.sin(np.linspace(0,40,tg.T))}
        op = portf.setup_optim_problem(timegrid = tg, prices = prices)
        res = op.optimize()
        out = eao.io.extract_output(portf = portf, op = op, res = res)
        # eao.io.output_to_file(out, 'test.xlsx')
        # get fill level from asset
        bat = portf.assets[0]
        fill_level = bat.fill_level(op, res)
        # calculate fill level from dispatch
        fill_level_check = -out['dispatch'].loc[:,'battery']
        fill_level_check[fill_level_check>0] *= eff
        fill_level_check = fill_level_check.cumsum()+50
        self.assertGreaterEqual(np.round(fill_level.min(), 3), 0)
        self.assertGreaterEqual(100, np.round(fill_level.max(), 3))
        self.assertAlmostEqual(abs(fill_level_check.values-fill_level).sum(), 0,4)
        # get fill level from output and check
        fl_out = out['internal_variables'].loc[:,'battery_fill_level'].values
        self.assertAlmostEqual(abs(fill_level-fl_out).sum(), 0,4)

    def test_battery_efficiency_asset_only(self):
        """ test specific setup of a battery
        """
        s = """{
            "__class__": "Portfolio",
            "assets": [
                {
                    "__class__": "Asset",
                    "asset_type": "Storage",
                    "block_size": null,
                    "cap_in": 50,
                    "cap_out": 50,
                    "cost_in": 0.0,
                    "cost_out": 0.0,
                    "cost_store": 0.0,
                    "eff_in": 0.9,
                    "end": null,
                    "end_level": 50.0,
                    "freq": null,
                    "inflow": 0.0,
                    "max_store_duration": null,
                    "name": "battery",
                    "no_simult_in_out": false,
                    "nodes": [
                        {
                            "__class__": "Node",
                            "commodity": null,
                            "name": "power",
                            "unit": {
                                "__class__": "Unit",
                                "factor": 1.0,
                                "flow": "MW",
                                "volume": "MWh"
                            }
                        }
                    ],
                    "periodicity": null,
                    "periodicity_duration": null,
                    "price": "price",
                    "profile": null,
                    "size": 100.0,
                    "start": null,
                    "start_level": 50.0,
                    "wacc": 0.0
                },
                {
                    "__class__": "Asset",
                    "asset_type": "SimpleContract",
                    "end": null,
                    "extra_costs": 0,
                    "freq": null,
                    "max_cap": 500,
                    "min_cap": -500,
                    "name": "supply",
                    "nodes": [
                        {
                            "__class__": "Node",
                            "commodity": null,
                            "name": "power",
                            "unit": {
                                "__class__": "Unit",
                                "factor": 1.0,
                                "flow": "MW",
                                "volume": "MWh"
                            }
                        }
                    ],
                    "periodicity": null,
                    "periodicity_duration": null,
                    "price": "price",
                    "profile": null,
                    "start": null,
                    "wacc": 0
                }
            ]
        }"""
        size = 100 # battery size
        eff = 0.9
        portf = eao.serialization.load_from_json(s)
        portf.assets[0].eff_in = eff
        portf.assets[0].size = size
        a = portf.assets[0]
        tg = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,10), freq = 'h')
        prices ={ 'price': np.sin(np.linspace(0,30,tg.T))}
        op = a.setup_optim_problem(timegrid = tg, prices = prices)
        res = op.optimize()
        # get fill level from asset
        fill_level_asset = a.fill_level(op, res)
        # calculate fill level from dispatch
        d_in  = -res.x[:tg.T]
        d_out = -res.x[tg.T:]
        fill_level = eff*d_in + d_out
        fill_level = fill_level.cumsum()+50
        self.assertGreaterEqual(fill_level.min(), 0)
        self.assertGreaterEqual(100, fill_level.max())
        self.assertAlmostEqual(abs(fill_level_asset-fill_level).sum(), 0,4)

    def test_opt_shortcut(self):
        """ test opt shortcut
        """
        ### manual benchmark
        node1 = eao.Node('node_1')
        node2 = eao.Node('node_2')
        timegrid = eao.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
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
        out = eao.io.extract_output(portf, op, res, prices)
        # shortcut
        #out2 = eao.io.optimize(portf, timegrid, prices)
        out2 = eao.optimize(portf, timegrid, prices)
        self.assertAlmostEqual(out['summary'].loc['value','Values'], out2['summary'].loc['value','Values'], 2)
        self.assertAlmostEqual(out['dispatch'].abs().sum().sum(), out2['dispatch'].abs().sum().sum(), 2)

    def test_opt_shortcut_split(self):
        """ test opt shortcut
        """
        ### manual benchmark
        s = """{
            "__class__": "Portfolio",
            "assets": [
                {
                    "__class__": "Asset",
                    "asset_type": "Storage",
                    "block_size": null,
                    "cap_in": 50,
                    "cap_out": 50,
                    "cost_in": 0.0,
                    "cost_out": 0.0,
                    "cost_store": 0.0,
                    "eff_in": 0.9,
                    "end": null,
                    "end_level": 50.0,
                    "freq": null,
                    "inflow": 0.0,
                    "max_store_duration": null,
                    "name": "battery",
                    "no_simult_in_out": false,
                    "nodes": [
                        {
                            "__class__": "Node",
                            "commodity": null,
                            "name": "power",
                            "unit": {
                                "__class__": "Unit",
                                "factor": 1.0,
                                "flow": "MW",
                                "volume": "MWh"
                            }
                        }
                    ],
                    "periodicity": null,
                    "periodicity_duration": null,
                    "price": null,
                    "profile": null,
                    "size": 100.0,
                    "start": null,
                    "start_level": 50.0,
                    "wacc": 0.0
                },
                {
                    "__class__": "Asset",
                    "asset_type": "SimpleContract",
                    "end": null,
                    "extra_costs": 0,
                    "freq": null,
                    "max_cap": 500,
                    "min_cap": -500,
                    "name": "supply",
                    "nodes": [
                        {
                            "__class__": "Node",
                            "commodity": null,
                            "name": "power",
                            "unit": {
                                "__class__": "Unit",
                                "factor": 1.0,
                                "flow": "MW",
                                "volume": "MWh"
                            }
                        }
                    ],
                    "periodicity": null,
                    "periodicity_duration": null,
                    "price": "price",
                    "profile": null,
                    "start": null,
                    "wacc": 0
                }
            ]
        }"""
        size = 100 # battery size
        eff = 1
        portf = eao.serialization.load_from_json(s)
        portf.assets[0].eff_in = eff
        portf.assets[0].size = size
        portf.assets[0].no_simult_in_out = False
        tg = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,3), freq = 'h')
        prices ={ 'price': np.sin(np.linspace(0,40,tg.T))}
        op = portf.setup_split_optim_problem(timegrid = tg, prices = prices, interval_size='d')
        res = op.optimize()
        out = eao.io.extract_output(portf = portf, op = op, res = res)
        # shortcut
        out2 = eao.optimize(portf, tg, prices, split_interval_size='d')
        self.assertAlmostEqual(out['summary'].loc['value','Values'], out2['summary'].loc['value','Values'], 2)
        self.assertAlmostEqual(out['dispatch'].abs().sum().sum(), out2['dispatch'].abs().sum().sum(), 2)

###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
