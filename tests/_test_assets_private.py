import unittest
import numpy as np
import pandas as pd
import datetime as dt
import json
from os.path import dirname, join
import sys

mypath = dirname(__file__)
test_data_path = join(mypath, "data", "CHP_asset")
sys.path.append(join(mypath, ".."))

import eaopack as eao


class PlantWithMarketRiskTestCase(unittest.TestCase):
    def test_simple_PP(self):
        """ Unit test. Setting up a CHPAsset with random prices
            and check that it generates full load at negative prices and nothing at positive prices.
        """
        ## baseline: with heat node, but no heat
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.CHPAsset(name='CHP', price='rand_price', nodes = (node_power, node_heat),
                                min_cap=5., max_cap=10.)
        np.random.seed(2709)
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op_o = a.setup_optim_problem(prices, timegrid=timegrid)
        res_o = op_o.optimize()
        x_power_o = np.around(res_o.x[:timegrid.T], decimals = 3) # round
        x_heat = np.around(res_o.x[timegrid.T:2*timegrid.T], decimals = 3) # round
        # heat or power / exchangable --> need to look at sum
        x_power_o += x_heat
        # self.assertTrue(all(x_heat==0))

        ## new: heat node is None
        a = eao.assets.PlantWithMarketRisk(name='CHP', price='rand_price', 
                                nodes = node_power,  # !!!!! heat node not given or None
                                min_cap=5., max_cap=10.)
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        x_power = np.around(res.x[:timegrid.T], decimals = 3) # round
        self.assertTrue(all(x_power==x_power_o))
    
    def test_PP_regression(self):
        """ Unit test. Predefined data - checking result is same as checked
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        node_gas = eao.assets.Node('node_gas')

        Start = dt.date(2022, 1, 1)
        End = dt.date(2022, 1, 3)
        timegrid = eao.assets.Timegrid(Start, End, freq='15min')

        #############################  test without heat node
        #####################################################
        # load test data
        myfile = join(test_data_path, "plant_test_data.csv")
        df = pd.read_csv(myfile)
        df.set_index('date', inplace = True)
        df = timegrid.prices_to_grid(df)
        # simple case, no min run time
        a = eao.assets.PlantWithMarketRisk(name='PP',
                                nodes=(node_power, node_gas),
                                min_cap         = 'mincap',
                                max_cap         = 'maxcap',
                                start_costs     = 1.,
                                running_costs   = 'runC',
                                fuel_efficiency = .5,
                                consumption_if_on= .1,
                                start_fuel      = 1,
                                min_downtime    = 2,
                                ramp            = 10,
                                time_already_running=0,
                                time_already_off= 1) 
        b = eao.assets.SimpleContract(name = 'powerMarket', price='power_price', nodes = node_power, min_cap=-100, max_cap=100)
        c = eao.assets.SimpleContract(name = 'gasMarket', price='gas_price', nodes = node_gas, min_cap=-100, max_cap=100)
        portf = eao.portfolio.Portfolio([a, b, c])
        op = portf.setup_optim_problem(df, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, df)
        ##### for manual check: eao.io.output_to_file(out, 'results_plant.xlsx')
        # # check manually checked values
        # check = out['prices']['PP (node_power)'].sum()
        self.assertAlmostEqual(res.value,  35926.718225, 2) 
        self.assertAlmostEqual(out['DCF'].sum().sum(),  35926.718225, 2)         
        self.assertAlmostEqual(out['dispatch'].sum().sum(),  0, 2)             
        # check = out['dispatch']['gasMarket (node_gas)'].sum()
        # self.assertAlmostEqual(check, 391.9 , 4)        

        # check serialization (new class...)
        s = eao.serialization.to_json(a)
        aa = eao.serialization.load_from_json(s)
        
    def test_PP_with_MR(self):
        """Unit test. Check costs allocation with threshold for power.
        """
        # Compare with normal plant
        node_power = eao.assets.Node('node_power')
        timegrid = eao.assets.Timegrid(dt.datetime(2021,1,1,0), dt.datetime(2021,1,1,3), freq = 'h')
        # Normal plant
        a = eao.assets.Plant(name='CHP', price='rand_price', 
                                nodes = node_power,
                                min_cap=5., max_cap=10.)
        np.random.seed(2709)
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5,
                 'second_price': np.ones(timegrid.T)*1}
        op_o = a.setup_optim_problem(prices, timegrid=timegrid)
        res_o = op_o.optimize()
        x_power_o = np.around(res_o.x[:timegrid.T], decimals = 3) # round

        ## new: plant with two variable costs
        # second price not active
        a = eao.assets.PlantWithMarketRisk(name='CHP', price='rand_price', 
                                nodes = node_power,
                                min_cap=1., max_cap=10.,
                                market_risk_threshold=10, # New part, no effect since threshold = max_cap
                                market_risk_costs='second_price') # New part
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()

        x_power = np.around(res.x[:timegrid.T], decimals = 3) # round
        self.assertTrue(all(x_power==x_power_o))
        
        ### Further test cases
        prices ={'rand_price': -np.ones(timegrid.T), # ones -> run with max_cap
                 'low_price': np.ones(timegrid.T)*0.1,
                 'high_price': np.ones(timegrid.T)*10}
        max_cap = 10.
        min_cap = 5.
        thres = 5.
        
        # Normal plant
        a = eao.assets.Plant(name='TestPlant', price='rand_price', 
                                nodes = node_power,
                                min_cap=min_cap, max_cap=max_cap)
        op_o = a.setup_optim_problem(prices, timegrid=timegrid)
        res_o = op_o.optimize()
        plant_power_o = np.around(res_o.x[:timegrid.T], decimals = 3) # round
        plant_power_calc = np.ones(timegrid.T)*max_cap
        plant_obj_o = res_o.value
        plant_obj_calc = -sum(prices['rand_price']*max_cap)
        
        self.assertTrue(all(plant_power_o==plant_power_calc)) # [10. 10. 10. 10. 10.]
        self.assertAlmostEqual(plant_obj_o, plant_obj_calc) # 50.0
        
        ## new: plant with two variable costs
        # second price always active
        a = eao.assets.PlantWithMarketRisk(name='TestPlantMR', price='rand_price', 
                                nodes = node_power,
                                min_cap=5., max_cap=10.,
                                market_risk_threshold=thres, # New part, effect all the time
                                market_risk_costs='low_price') # New part
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        plant_MR_power_o = np.around(res.x[:timegrid.T], decimals = 3) # round
        plant_MR_power_calc = np.ones(timegrid.T)*max_cap
        plant_MR_obj_o = res.value
        plant_MR_obj_calc = -sum(prices['rand_price']*max_cap + prices['low_price']*(max_cap-thres)) # low_price * h_above
        
        self.assertTrue(all(plant_MR_power_o==plant_MR_power_calc)) # [10. 10. 10. 10. 10.]
        self.assertAlmostEqual(plant_MR_obj_o, plant_MR_obj_calc) # 47.5
        
        ## new: plant with two variable costs
        # high second price -> dispatch below threshold
        thres = 8.
        a = eao.assets.PlantWithMarketRisk(name='TestPlantMR', price='rand_price', 
                                nodes = node_power,
                                min_cap=5., max_cap=10.,
                                market_risk_threshold=thres, # New part, effect all the time
                                market_risk_costs='high_price') # New part
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        plant_MR_power_o = np.around(res.x[:timegrid.T], decimals = 3) # round
        plant_MR_power_calc = np.ones(timegrid.T)*thres # should be at threshold
        plant_MR_obj_o = res.value
        plant_MR_obj_calc = -sum(prices['rand_price']*thres + prices['high_price']*(thres-thres)) # high_price * h_above approx 0
        
        self.assertTrue(all(plant_MR_power_o==plant_MR_power_calc)) # [8. 8. 8. 8. 8.]
        self.assertAlmostEqual(plant_MR_obj_o, plant_MR_obj_calc) # 40.0

class PlantWithStepStartupCostTestCase(unittest.TestCase):
    def test_simple_PP(self):
        """ Unit test. Setting up a CHPAsset with random prices
            and check that it generates full load at negative prices and nothing at positive prices.
        """
        ## baseline: with heat node, but no heat
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.CHPAsset(name='CHP', price='rand_price', nodes = (node_power, node_heat),
                                min_cap=5., max_cap=10.)
        np.random.seed(2709)
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op_o = a.setup_optim_problem(prices, timegrid=timegrid)
        res_o = op_o.optimize()

        x_power_o = np.around(res_o.x[:timegrid.T], decimals = 3) # round
        x_heat = np.around(res_o.x[timegrid.T:2*timegrid.T], decimals = 3) # round
        # heat or power / exchangable --> need to look at sum
        x_power_o += x_heat
        # self.assertTrue(all(x_heat==0))

        ## new: heat node is None
        a = eao.assets.PlantWithStepStartupCost(name='coal_plant', price='rand_price', 
                                nodes = node_power,
                                min_cap=5., max_cap=10.,
                                rampUp_Load_1 = 11., # New part
                                rampUp_Load_2 = 11., # Higher than max_cap -> constrains inactive 
                                rampUp_Load_3 = 11., 
                                rampUp_Cost_1 = 0.,
                                rampUp_Cost_2 = 0., 
                                rampUp_Cost_3 = 0.,
                                last_dispatch = 0.)
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        x_power = np.around(res.x[:timegrid.T], decimals = 3) # round
        self.assertTrue(all(x_power==x_power_o))
        
    def test_PP_with_step_startup_costs(self):
        """Unit test. Check costs allocation with step start-up costs.
        """
        # Compare with normal plant
        node_power = eao.assets.Node('node_power')
        timegrid = eao.assets.Timegrid(dt.datetime(2021,1,1,0), dt.datetime(2021,1,1,3), freq = 'h')
        # Normal plant
        a = eao.assets.Plant(name='plant', price='rand_price', 
                                nodes = node_power,
                                min_cap=1., max_cap=10.,
                                last_dispatch = 0.)
        np.random.seed(2709)
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op_o = a.setup_optim_problem(prices, timegrid=timegrid)
        res_o = op_o.optimize()
        x_power_o = np.around(res_o.x[:timegrid.T], decimals = 3) # round
        value_0 = np.round(res_o.value, decimals = 2)

        ## new: plant with step start costs
        # startcosts not active -> Same result expected
        a = eao.assets.PlantWithStepStartupCost(
                                name='coal_plant', 
                                price='rand_price', 
                                nodes = node_power,
                                min_cap=1., max_cap=10.,
                                rampUp_Load_1 = 2., # New part
                                rampUp_Load_2 = 4., # New part
                                rampUp_Load_3 = 6., # New part
                                rampUp_Cost_1 = 0., # New part 
                                rampUp_Cost_2 = 0., # New part
                                rampUp_Cost_3 = 0.,
                                last_dispatch = 0.) # New part
        
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        
        x_power = np.around(res.x[:timegrid.T], decimals = 3) # round
        self.assertTrue(all(x_power==x_power_o)) # Check if disptach is unchanged
        self.assertTrue(all(res.x[3:14] == 1)) # Check if on variables are equal to 1
        
        # Test specifically for first start variable
        a = eao.assets.PlantWithStepStartupCost(
                                name='coal_plant',
                                price='rand_price', 
                                nodes = node_power,
                                min_cap=1., max_cap=10.,
                                rampUp_Load_1 = 2., # Only bool_on_1 should be on
                                rampUp_Load_2 = 11., # Higher than max_cap -> should be off
                                rampUp_Load_3 = 11.,
                                rampUp_Cost_1 = 0.1, 
                                rampUp_Cost_2 = 0.1,
                                rampUp_Cost_3 = 0.1,
                                last_dispatch = 0.)
        
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        value_adj = np.round(res.value + 0.1, decimals = 2)
        
        self.assertTrue(all(res.x[0:2] == 10)) # Disp unchanged
        self.assertTrue(all(res.x[3:8] == 1)) # Only bool_on_1 and CHP should be on 
        self.assertTrue(all(res.x[9:14] == 0)) # Other on variables should be off
        self.assertTrue(res.x[15] == 1) # bool_start_1 should be 1, at beginning t = 0
        self.assertTrue(all(res.x[16:] == 0)) # since they have cost and are not needed, other start variabels should be 0
        self.assertTrue(value_0 == value_adj) # Check obj function value
        
        # Test if last dispatch was not 0
        # Test specifically for first start variable
        a = eao.assets.PlantWithStepStartupCost(
                                name='coal_plant', 
                                price='rand_price', 
                                nodes = node_power,
                                min_cap=1., max_cap=10.,
                                rampUp_Load_1 = 2.,
                                rampUp_Load_2 = 4.,
                                rampUp_Load_3 = 6.,
                                rampUp_Cost_1 = 0.1, 
                                rampUp_Cost_2 = 0.1,
                                rampUp_Cost_3 = 0.1,
                                last_dispatch = 8.)
        
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        
        self.assertTrue(all(res.x[0:2] == 10)) # Disp unchanged
        self.assertTrue(all(res.x[3:14] == 1)) # All bool on should be 1 
        self.assertTrue(all(res.x[15:] == 0)) # bool_start_1 and other start variables should be 0 (since last_disp > 0)
        
        # Test for other start variales
        a = eao.assets.PlantWithStepStartupCost(
                                name='coal_plant', 
                                price='rand_price', 
                                nodes = node_power,
                                min_cap=1., max_cap=10.,
                                rampUp_Load_1 = 2.,
                                rampUp_Load_2 = 4.,
                                rampUp_Load_3 = 6.,
                                rampUp_Cost_1 = 0.1, 
                                rampUp_Cost_2 = 0.1,
                                rampUp_Cost_3 = 0.1,
                                last_dispatch = 0.)
        
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        value_adj_2 = np.round(res.value + 0.3, decimals = 2)
        
        self.assertTrue(all(res.x[0:2] == 10)) # Disp unchanged
        self.assertTrue(all(res.x[3:14] == 1)) # All on variables should be on 
        self.assertTrue(res.x[15] == 1) # bool_start_1 should be 1, at beginning t = 0
        self.assertTrue(res.x[18] == 1) # bool_start_2 should be 1, at beginning t = 0
        self.assertTrue(res.x[21] == 1) # bool_start_3 should be 1, at beginning t = 0
        self.assertTrue(all(res.x[16:17] == 0)) # since they have cost and are not needed, other start variabels should be 0
        self.assertTrue(all(res.x[19:20] == 0))
        self.assertTrue(all(res.x[22:23] == 0))
        self.assertTrue(value_0 == value_adj_2) # Check obj function value
        
        # Test with two starts
        prices ={'rand_price': np.array([-1, -1, 1, 1, -1, -1])}
        timegrid = eao.assets.Timegrid(dt.datetime(2021,1,1,0), dt.datetime(2021,1,1,6), freq = 'h')
        
        a = eao.assets.PlantWithStepStartupCost(
                                name='coal_plant', 
                                price='rand_price', 
                                nodes = node_power,
                                min_cap=1., max_cap=10.,
                                rampUp_Load_1 = 2., # Only bool_on_1 should be on
                                rampUp_Load_2 = 4., # Higher than max_cap -> should be off
                                rampUp_Load_3 = 6.,
                                rampUp_Cost_1 = 0.1, 
                                rampUp_Cost_2 = 0.1,
                                rampUp_Cost_3 = 0.1,
                                last_dispatch = 0.)
        
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertTrue(all(res.x[0:1] == 10)) # Full disp fist two hours
        self.assertTrue(all(res.x[2:3] == 0)) # No disp at negative prices
        self.assertTrue(all(res.x[4:5] == 10)) # Full disp last two hours
        self.assertTrue(all(res.x[6:7] == 1)) # Bool on accordingly
        self.assertTrue(all(res.x[8:9] == 0)) # Bool on accordingly
        self.assertTrue(all(res.x[10:11] == 1)) # Bool on accordingly
        
        self.assertTrue(all(res.x[12:13] == 1)) # Same for other bool on variables
        self.assertTrue(all(res.x[14:15] == 0)) # Same for other bool on variables
        self.assertTrue(all(res.x[16:19] == 1)) # Same for other bool on variables
        self.assertTrue(all(res.x[20:21] == 0)) # Same for other bool on variables
        self.assertTrue(all(res.x[22:25] == 1)) # Same for other bool on variables
        self.assertTrue(all(res.x[26:27] == 0)) # Same for other bool on variables
        self.assertTrue(all(res.x[28:29] == 1)) # Same for other bool on variables

        self.assertTrue(res.x[30] == 1) # Start at beginning
        self.assertTrue(res.x[34] == 1) # Start at step 4
        self.assertTrue(res.x[36] == 1) # Start at beginning
        self.assertTrue(res.x[40] == 1) # Start at step 4
        self.assertTrue(res.x[42] == 1) # Start at beginning
        self.assertTrue(res.x[46] == 1) # Start at step 4        
        self.assertTrue(all(res.x[31:33] == 0)) # Off otherwise
        self.assertTrue(res.x[35] == 0) # Off otherwise
        self.assertTrue(all(res.x[37:39] == 0)) # Off otherwise
        self.assertTrue(res.x[41] == 0) # Off otherwise
        self.assertTrue(all(res.x[43:45] == 0)) # Off otherwise
        self.assertTrue(res.x[47] == 0) # Off otherwise

    def test_quarter_hours(self):
        """Unit test. Test behaviour with quarter-hour time intervalls.
        """
        # Compare with normal plant
        node_power = eao.assets.Node('node_power')
        timegrid = eao.assets.Timegrid(dt.datetime(2021,1,1,0), dt.datetime(2021,1,1,3), freq = '15min')
        
        # Prices
        np.random.seed(2709)
        prices ={'rand_price': np.ones(timegrid.T)*-1.0}
        
        # Test value function
        # No rampUp_costs
        a = eao.assets.PlantWithStepStartupCost(
                                name='coal_plant',
                                price='rand_price', 
                                nodes = node_power,
                                min_cap=1., max_cap=10.,
                                rampUp_Load_1 = 2., 
                                rampUp_Load_2 = 4.,
                                rampUp_Load_3 = 6.,
                                rampUp_Cost_1 = 0.0, 
                                rampUp_Cost_2 = 0.0,
                                rampUp_Cost_3 = 0.0,
                                last_dispatch = 0.)
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        value_0 = np.round(res.value, decimals = 2)
        
        # With rampUp_costs
        a = eao.assets.PlantWithStepStartupCost(
                                name='coal_plant',
                                price='rand_price', 
                                nodes = node_power,
                                min_cap=1., max_cap=10.,
                                rampUp_Load_1 = 2., 
                                rampUp_Load_2 = 4.,
                                rampUp_Load_3 = 6.,
                                rampUp_Cost_1 = 0.1, 
                                rampUp_Cost_2 = 0.1,
                                rampUp_Cost_3 = 0.1,
                                last_dispatch = 0.)
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        value_adj = np.round(res.value + 0.3, decimals = 2)
        
        self.assertTrue(all(res.x[12:23] == 1)) # All on variables should be on 
        self.assertTrue(res.x[60] == 1) # bool_start_1 should be 1, at beginning t = 0
        self.assertTrue(res.x[72] == 1) # bool_start_2 should be 1, at beginning t = 0
        self.assertTrue(res.x[84] == 1) # bool_start_3 should be 1, at beginning t = 0
        self.assertTrue(value_0 == value_adj) # Check obj function value



class StorageClientTest(unittest.TestCase):
    def test_soc_max(self):
        """ trivial test with eff_out
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,2), freq = 'h')
        a = eao.assets.StorageClient('STORAGE', node,
                               size=5,
                               cap_in=1,
                               cap_out=1,
                               soc_max=4,
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
        self.assertAlmostEqual(fl.max(), 4, 5)
        print(res)


    def test_soc_min(self):
        """ trivial test with eff_out
        """
        node = eao.assets.Node('testNode')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,2), freq = 'h')
        a = eao.assets.StorageClient('STORAGE', node,
                               size=5,
                               cap_in=1,
                               cap_out=1,
                               soc_min='soc_min',
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

        soc_min = np.ones([timegrid.T]) * 1
        soc_min[:10] = 0

        prices ={ 'price': price, 'soc_min': soc_min}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        xin = res.x[0:24]
        xout = res.x[24:48]
        fl = a.fill_level(op, res)
        self.assertAlmostEqual(-xin.sum()/xout.sum(), 1/.9/.8, 3) # overall loss
        self.assertAlmostEqual(fl.max(), 5, 5)
        print(res)