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

class CHPAssetTest(unittest.TestCase):
    def test_optimization(self):
        """ Unit test. Setting up a CHPAsset with random prices
            and check that it generates full load at negative prices and nothing at positive prices.
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.CHPAsset(name='CHP', price='rand_price', nodes = (node_power, node_heat),
                                min_cap=5., max_cap=10.)
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        x_power = np.around(res.x[:timegrid.T], decimals = 3) # round
        x_heat = np.around(res.x[timegrid.T:2*timegrid.T], decimals = 3) # round

        check =     all(x_power[np.sign(op.c[:timegrid.T]) == -1] + a.conversion_factor_power_heat * x_heat[np.sign(op.c[:timegrid.T]) == -1] == op.u[:timegrid.T][np.sign(op.c[:timegrid.T]) == -1]) \
                and all(x_power[np.sign(op.c[:timegrid.T]) == 1] + a.conversion_factor_power_heat * x_heat[np.sign(op.c[:timegrid.T]) == 1] == 0)
        tot_dcf = np.around((a.dcf(op, res)).sum(), decimals = 3) # asset dcf, calculated independently
        check = check and (tot_dcf == np.around(res.value , decimals = 3))
        self.assertTrue(check)

    def test_min_cap_vector(self):
        """ Unit test. Setting up a CHPAsset with positive prices and a simple contract with a minimum demand that is
            smaller than the min capacity. Check that it runs at minimum capacity.
        """
        np.random.seed(42)

        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 10)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        # capacities
        restr_times = pd.date_range(Start, End, freq='d', inclusive='left')
        min_cap = {}
        min_cap['start'] = restr_times.to_list()
        min_cap['end'] = (restr_times + dt.timedelta(days=1)).to_list()
        min_cap['values'] = np.random.rand(len(min_cap['start']))
        max_cap = 1.

        max_cap_sc = min_cap.copy()
        max_cap_sc['values'] = -0.5*min_cap['values']

        sc_power = eao.assets.SimpleContract(name='SC_power', price='rand_price', nodes=node_power,
                                            min_cap=-20., max_cap=max_cap_sc)

        a = eao.assets.CHPAsset(name='CHP', nodes=(node_power, node_heat),
                                min_cap=min_cap, max_cap=max_cap)

        prices = {'rand_price': -np.ones(timegrid.T)}
        p = eao.portfolio.Portfolio([a, sc_power])

        op = p.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        x_power = np.around(res.x[:timegrid.T], decimals=3)  # round
        x_heat = np.around(res.x[timegrid.T:2 * timegrid.T], decimals=3)  # round

        self.assertAlmostEqual(np.abs(timegrid.values_to_grid(min_cap) - x_power + a.conversion_factor_power_heat * x_heat).max(), 0., 2)

    def test_max_cap_vector(self):
        """ Unit test. Setting up a CHPAsset with negative prices
            and check that it runs at maximum capacity.
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 10)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        # capacities
        restr_times = pd.date_range(Start, End, freq='d', inclusive='left')
        min_cap = {}
        min_cap['start'] = restr_times.to_list()
        min_cap['end'] = (restr_times + dt.timedelta(days=1)).to_list()
        min_cap['values'] = np.random.rand(len(min_cap['start']))
        max_cap = min_cap.copy()

        a = eao.assets.CHPAsset(name='CHP', price='rand_price', nodes=(node_power, node_heat),
                                min_cap=min_cap, max_cap=max_cap)

        prices = {'rand_price': -np.ones(timegrid.T)}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        self.assertAlmostEqual(np.abs(res.x[:timegrid.T] + a.conversion_factor_power_heat * res.x[timegrid.T:2 * timegrid.T] - timegrid.values_to_grid(max_cap)).sum(), 0., 5)

    def test_start_variables(self):
        """ Unit test. Setting up a CHPAsset with random prices
            and check that it starts any time the prices change from positive to negative.
        """
        np.random.seed(42)
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 10)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        a = eao.assets.CHPAsset(name='CHP', price='rand_price', nodes=(node_power, node_heat), min_cap=0.001, max_cap=1, start_costs=0.0001, running_costs=0)
        prices ={'rand_price': np.random.randint(low=-1, high=2, size=timegrid.T)}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        start_variables = res.x[3*timegrid.T:]
        for i in range(1, timegrid.T):
            if np.sign(op.c[i]) == -1 and np.sign(np.sign(op.c[i-1])) == 1:
                self.assertTrue(start_variables[i]==1)

    def test_minruntime(self):
        """ Unit test. Setting up a CHPAsset and check min run time restriction
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 2)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        # simple case, no min run time
        a = eao.assets.CHPAsset(name='CHP',
                                price='price',
                                nodes=(node_power, node_heat),
                                min_cap=1.,
                                max_cap=10.,
                                start_costs=1.,
                                running_costs=5.)
        prices ={'price': 1.*np.ones(timegrid.T)}
        prices['price'][0:10] = -100.

        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        on_variables = res.x[2*timegrid.T:3*timegrid.T]
        disp_variables = res.x[0*timegrid.T:1*timegrid.T]
        start_variables = res.x[3*timegrid.T:]
        self.assertAlmostEqual(res.value, 10*1000. + 10*(-5)-1., 4)
        # min run time 20
        a = eao.assets.CHPAsset(name='CHP',
                                price='price',
                                nodes=(node_power, node_heat),
                                min_cap=1.,
                                max_cap=10.,
                                start_costs=1.,
                                running_costs=5.,
                                min_runtime=20)
        prices ={'price': 1.*np.ones(timegrid.T)}
        prices['price'][0:10] = -100.

        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        on_variables = res.x[2*timegrid.T:3*timegrid.T]
        disp_variables = res.x[0*timegrid.T:1*timegrid.T]
        start_variables = res.x[3*timegrid.T:]
        self.assertAlmostEqual(on_variables.sum(), 20., 4)
        # 10 times full load, 10 time min load
        self.assertAlmostEqual(res.value, 10*10*100. - 10*1 + 20*(-5)-1., 4)
        # min run time 20 ... but 5 hours already on
        a = eao.assets.CHPAsset(name='CHP',
                                price='price',
                                nodes=(node_power, node_heat),
                                min_cap=1.,
                                max_cap=10.,
                                start_costs=1.,
                                running_costs=5.,
                                min_runtime=20,
                                time_already_running=5)
        prices ={'price': 1.*np.ones(timegrid.T)}
        prices['price'][0:10] = -100.

        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        on_variables = res.x[2*timegrid.T:3*timegrid.T]
        disp_variables = res.x[0*timegrid.T:1*timegrid.T]
        start_variables = res.x[3*timegrid.T:]
        self.assertAlmostEqual(on_variables.sum(), 15., 4)
        # 10 times full load, 10 time min load, NO start!
        self.assertAlmostEqual(res.value, 10*10*100. - 5*1 + 15*(-5), 4)

    def test_mindowntime(self):
        """ Unit test. Setting up a CHPAsset and check min down time restriction
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 2)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        a = eao.assets.CHPAsset(name='CHP',
                                price='price',
                                nodes=(node_power, node_heat),
                                min_cap=1.,
                                max_cap=10.,
                                min_downtime=5,
                                time_already_off=1)
        prices = {'price': -1. * np.ones(timegrid.T)}
        prices['price'][10] = 100

        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        on_variables = res.x[2 * timegrid.T:3 * timegrid.T]
        self.assertAlmostEqual(on_variables.sum(), timegrid.T-9, 4)
        # Asset is off for the first 4 timesteps, and off again for min_downtime=5 timesteps around hour 10 when price
        # is 100, otherwise on because price is negative
        # T-9 times full load at price -1
        self.assertAlmostEqual(res.value, (timegrid.T-9)*10, 4)

    def test_gas_consumption(self):
        """ Unit test. Setting up a CHPAsset with explicit gas (fuel) consumption
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        node_gas = eao.assets.Node('node_gas')

        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 2)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        # simple case, no min run time
        a = eao.assets.CHPAsset(name='CHP',
                                nodes=(node_power, node_heat, node_gas),
                                min_cap=1.,
                                max_cap=10.,
                                start_costs=1.,
                                running_costs=5.,
                                conversion_factor_power_heat= 0.2,
                                max_share_heat= 1,
                                start_fuel = 10,
                                fuel_efficiency= .5,
                                consumption_if_on= .1)
        b = eao.assets.SimpleContract(name = 'powerMarket', price='price', nodes = node_power, min_cap=-100, max_cap=100)
        c = eao.assets.SimpleContract(name = 'gasMarket', price='priceGas', nodes = node_gas, min_cap=-100, max_cap=100)
        d = eao.assets.SimpleContract(name = 'heatMarket', price='priceGas', nodes = node_heat, min_cap=-100, max_cap=100)
        prices ={'price': 50.*np.ones(timegrid.T), 'priceGas': 0.1*np.ones(timegrid.T)}
        prices['price'][0:5] = -100.
        portf = eao.portfolio.Portfolio([a, b, c, d])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        # check manually checked values
        check = out['dispatch']['CHP (node_power)'].sum()
        self.assertAlmostEqual(check, 190. , 4)
        check = out['dispatch']['CHP (node_heat)'].sum()
        self.assertAlmostEqual(check, 0. , 4)
        check = out['dispatch']['gasMarket (node_gas)'].sum()
        self.assertAlmostEqual(check, 391.9 , 4)

    def test_min_take(self):
        """
        Unit test. Set up a CHPAsset with positive prices and check that the minTake is reached in the defined time frame.
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 10)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        min_take = {}
        min_take['start'] = dt.date(2021, 1, 3)
        min_take['end'] = dt.date(2021, 1, 5)
        min_take['values'] = 20
        # max_take = min_take.copy()
        max_take = None
        conversion_factor_power_heat = 0.5

        a = eao.assets.CHPAsset(name='CHP', price='rand_price', nodes=(node_power, node_heat), min_cap=0., max_cap=20,
                                min_take=min_take, max_take=max_take, conversion_factor_power_heat=conversion_factor_power_heat)
        prices = {'rand_price': np.ones(timegrid.T)}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        disp_power = res.x[:timegrid.T]
        disp_heat = res.x[timegrid.T:2*timegrid.T]

        self.assertAlmostEqual((disp_power[2*24:4*24]+conversion_factor_power_heat*disp_heat[2*24:4*24]).sum(), 20, 5)

    def test_max_take(self):
        """
        Unit test. Set up a CHPAsset with negative prices and check that the max_Take is reached in the defined time frame.
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 10)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        max_take = {}
        max_take['start'] = dt.date(2021, 1, 3)
        max_take['end'] = dt.date(2021, 1, 5)
        max_take['values'] = 20
        # max_take = min_take.copy()
        min_take = None
        conversion_factor_power_heat = 0.5

        a = eao.assets.CHPAsset(name='CHP', price='rand_price', nodes=(node_power, node_heat), min_cap=0., max_cap=20,
                                min_take=min_take, max_take=max_take, conversion_factor_power_heat=conversion_factor_power_heat)

        # test serialization
        s = eao.serialization.to_json(a)
        aa = eao.serialization.load_from_json(s)

        prices = {'rand_price': -np.ones(timegrid.T)}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        disp_power = res.x[:timegrid.T]
        disp_heat = res.x[timegrid.T:2*timegrid.T]

        self.assertAlmostEqual((disp_power[2*24:4*24]+conversion_factor_power_heat*disp_heat[2*24:4*24]).sum(), 20, 5)

    def test_freq_conversion(self):
        """
        Check the time conversion function that is used to convert e.g. min_runtime and time_already_running in CHPAsset
        """
        old_value = 10 * 30
        new_value = eao.basic_classes.convert_time_unit(old_value, old_freq='h', new_freq='d')
        self.assertAlmostEqual(new_value - old_value / 24, 0, 5)

        old_value = 12 * 30
        new_value = eao.basic_classes.convert_time_unit(old_value, old_freq='h', new_freq='15min')
        self.assertAlmostEqual(new_value - old_value * 4, 0, 5)

        old_value = 14 * 30
        new_value = eao.basic_classes.convert_time_unit(old_value, old_freq='d', new_freq='15min')
        self.assertAlmostEqual(new_value - old_value * 24 * 4, 0, 5)

        old_value = 16 * 30
        new_value = eao.basic_classes.convert_time_unit(old_value, old_freq='30min', new_freq='min')
        self.assertAlmostEqual(new_value - old_value * 30, 0, 5)

        old_value = 22 * 30
        new_value = eao.basic_classes.convert_time_unit(old_value, old_freq='min', new_freq='h')
        self.assertAlmostEqual(new_value - old_value / 60, 0, 5)

    def test_freq_conversion2(self):
        """ Unit test. Setting up two CHP Assets with different freq and checking that both give the same results
        """
        np.random.seed(42)

        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        node_fuel = eao.assets.Node('node_fuel')

        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 2)
        prices = {}

        timegrid1 = eao.assets.Timegrid(Start, End, freq='h', main_time_unit='h')
        a1 = eao.assets.CHPAsset(name='CHP',
                                 price='price1',
                                 nodes=(node_power, node_heat, node_fuel),
                                 extra_costs=1,
                                 min_cap=5,
                                 max_cap=20,
                                 conversion_factor_power_heat=0.5,
                                 max_share_heat=0.8,
                                 # ramp=6,
                                 start_costs=0.4,
                                 running_costs=0.3,
                                 min_runtime=4,
                                 time_already_running=0,
                                 min_downtime=2,
                                 time_already_off=1,
                                 last_dispatch=0,
                                 start_fuel=0.7,
                                 fuel_efficiency=0.8,
                                 consumption_if_on=0.2
                                 )
        price1 = 20 * np.random.rand(timegrid1.T) - 10
        prices['price1'] = price1
        op1 = a1.setup_optim_problem(prices, timegrid=timegrid1)
        res1 = op1.optimize()

        timegrid2 = eao.assets.Timegrid(Start, End, freq='15min', main_time_unit='h')
        a2 = eao.assets.CHPAsset(name='CHP',
                                 price='price2',
                                 nodes=(node_power, node_heat, node_fuel),
                                 extra_costs=1,
                                 min_cap=5,
                                 max_cap=20,
                                 conversion_factor_power_heat=0.5,
                                 max_share_heat=0.8,
                                 # ramp=6,
                                 start_costs=0.4,
                                 running_costs=0.3,
                                 min_runtime=4,
                                 time_already_running=0,
                                 min_downtime=2,
                                 time_already_off=1,
                                 last_dispatch=0,
                                 start_fuel=0.7,
                                 fuel_efficiency=0.8,
                                 consumption_if_on=0.2
                                 )
        price2 = np.vstack([price1] * 4).T.reshape(-1)
        prices['price2'] = price2
        op2 = a2.setup_optim_problem(prices, timegrid=timegrid2)
        res2 = op2.optimize()

        self.assertAlmostEqual(res1.value, res2.value, 4)

    def test_start_and_shutdown_ramp1(self):
        """ Unit test. Setting up a CHPAsset and checking start and shutdown ramps where the upper and lower bounds are equal
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 2)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        a = eao.assets.CHPAsset(name='CHP',
                                price='price',
                                nodes=(node_power, node_heat),
                                min_cap=5.,
                                max_cap=10.,
                                time_already_off=1,
                                start_ramp_lower_bounds=[1, 2, 3, 4],
                                shutdown_ramp_lower_bounds=[1, 4, 6, 7],
                                time_already_running=2)
        prices = {'price': -1. * np.ones(timegrid.T)}
        prices['price'][15:] = 100

        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        # Asset starts out with the last 2 steps of the startramp, then runs at maximum capacity for 9 steps and then
        # follows the shutdown ramp
        self.assertAlmostEqual(res.value, 3 + 4 + 9 * 10 + 7 + 6 + 4 + 1, 4)

    def test_start_and_shutdown_ramp2(self):
        """ Unit test. Setting up a CHPAsset and checking start and shutdown ramps where the upper and lower bounds are different
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 2)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        start_ramp_lower_bounds = [1, 2, 3, 4]
        start_ramp_upper_bounds = [2, 3, 5, 5]
        shutdown_ramp_lower_bounds = [1, 4, 4.5, 5, 5]
        shutdown_ramp_upper_bounds = [2, 4, 5, 8, 10]
        min_cap=5
        max_cap=10
        min_runtime = 10

        # Case use all lower bounds:
        a = eao.assets.CHPAsset(name='CHP',
                                price='price',
                                nodes=(node_power, node_heat),
                                min_cap=min_cap,
                                max_cap=max_cap,
                                time_already_off=1,
                                start_ramp_lower_bounds=start_ramp_lower_bounds,
                                start_ramp_upper_bounds=start_ramp_upper_bounds,
                                shutdown_ramp_lower_bounds=shutdown_ramp_lower_bounds,
                                shutdown_ramp_upper_bounds=shutdown_ramp_upper_bounds,
                                time_already_running=1,
                                min_runtime=min_runtime)
        prices = {'price': 1. * np.ones(timegrid.T)}

        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        power_variables = res.x[0 * timegrid.T:1 * timegrid.T]
        heat_variables = res.x[1 * timegrid.T:2 * timegrid.T]
        on_variables = res.x[2 * timegrid.T:3 * timegrid.T]
        start_variables = res.x[3 * timegrid.T:4 * timegrid.T]
        shutdown_variables = res.x[4 * timegrid.T:]
        # Asset starts out with the last 3 steps of the startramp, then runs at maximum capacity for min_runtime steps and then
        # follows the shutdown ramp
        disp_res = power_variables + a.conversion_factor_power_heat * heat_variables
        disp_true = start_ramp_lower_bounds[1:] + min_runtime * [min_cap] + list(reversed(shutdown_ramp_lower_bounds)) + [0.] * 6
        self.assertAlmostEqual(abs(disp_res - disp_true).sum(), 0, 4)
        start_variables_true = np.zeros(timegrid.T)
        self.assertAlmostEqual(abs(start_variables_true - start_variables).sum(), 0, 4)
        shutdown_variables_true = np.zeros(timegrid.T)
        shutdown_variables_true[len(start_ramp_lower_bounds) - 1 + min_runtime + len(shutdown_ramp_lower_bounds)] = 1
        self.assertAlmostEqual(abs(shutdown_variables_true - shutdown_variables).sum(), 0, 4)

        # Case use all upper bounds
        a = eao.assets.CHPAsset(name='CHP',
                                price='price',
                                nodes=(node_power, node_heat),
                                min_cap=min_cap,
                                max_cap=max_cap,
                                start_ramp_lower_bounds=start_ramp_lower_bounds,
                                start_ramp_upper_bounds=start_ramp_upper_bounds,
                                shutdown_ramp_lower_bounds=shutdown_ramp_lower_bounds,
                                shutdown_ramp_upper_bounds=shutdown_ramp_upper_bounds)
        prices = {'price': 10 * np.ones(timegrid.T)}
        prices['price'][5: 5 + len(start_ramp_lower_bounds)+ len(shutdown_ramp_lower_bounds)]=-1

        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        power_variables = res.x[0 * timegrid.T:1 * timegrid.T]
        heat_variables = res.x[1 * timegrid.T:2 * timegrid.T]
        on_variables = res.x[2 * timegrid.T:3 * timegrid.T]
        start_variables = res.x[3 * timegrid.T:4 * timegrid.T]
        shutdown_variables = res.x[4 * timegrid.T:]
        # Asset follows start ramp, then immediately shutdown ramp at highest possible load while the price is
        # negative, otherwise asset is off
        disp_res = power_variables + a.conversion_factor_power_heat * heat_variables
        disp_true =[0.] * 5 + start_ramp_upper_bounds + list(reversed(shutdown_ramp_upper_bounds)) + [0.] * (timegrid.T-len(start_ramp_lower_bounds)-len(shutdown_ramp_lower_bounds)-5)
        self.assertAlmostEqual(abs(disp_res - disp_true).sum(), 0, 4)
        start_variables_true = np.zeros(timegrid.T)
        start_variables_true[5] = 1
        self.assertAlmostEqual(abs(start_variables_true - start_variables).sum(), 0, 4)
        shutdown_variables_true = np.zeros(timegrid.T)
        shutdown_variables_true[5 + len(start_ramp_lower_bounds)+ len(shutdown_ramp_lower_bounds)] = 1
        self.assertAlmostEqual(abs(shutdown_variables_true - shutdown_variables).sum(), 0, 4)

    def test_start_and_shutdown_ramp3(self):
        """ Unit test. Setting up a CHPAsset and checking interpolation of start and shutdown ramps when timegrid.freq
            and timegrid.main_time_unit are different
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 2)
        timegrid = eao.assets.Timegrid(Start, End, freq='h', main_time_unit='h')

        start_ramp = [1, 2, 3, 4]
        shutdown_ramp = [1, 3, 5, 5, 5]
        min_cap = 5
        max_cap = 10
        ramp_freq = '15min'

        a = eao.assets.CHPAsset(name='CHP',
                                price='price',
                                nodes=(node_power, node_heat),
                                min_cap=min_cap,
                                max_cap=max_cap,
                                start_ramp_lower_bounds=start_ramp,
                                shutdown_ramp_lower_bounds=shutdown_ramp,
                                ramp_freq=ramp_freq
                                )
        prices = {'price': 10 * np.ones(timegrid.T)}
        prices['price'][5: 5 + int(np.ceil(len(start_ramp)/4)+ np.ceil(len(shutdown_ramp)/4))]=-1

        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        power_variables = res.x[0 * timegrid.T:1 * timegrid.T]
        heat_variables = res.x[1 * timegrid.T:2 * timegrid.T]
        on_variables = res.x[2 * timegrid.T:3 * timegrid.T]
        start_variables = res.x[3 * timegrid.T:4 * timegrid.T]
        shutdown_variables = res.x[4 * timegrid.T:]
        # Asset follows interpolated start ramp, then immediately interpolated shutdown ramp at highest possible load while the price is
        # negative, otherwise asset is off
        converted_time_rounded = int(np.ceil(eao.basic_classes.convert_time_unit(value=1, old_freq=timegrid.freq, new_freq=ramp_freq)))
        start_ramp_padding_size = int(np.ceil(len(start_ramp) / converted_time_rounded) * converted_time_rounded) - len(start_ramp)
        start_ramp_padded = start_ramp + [start_ramp[-1]] * start_ramp_padding_size
        start_ramp_interpolated = np.reshape(start_ramp_padded, (-1, converted_time_rounded)).mean(axis=1)

        shutdown_ramp_padding_size = int(np.ceil(len(shutdown_ramp) / converted_time_rounded) * converted_time_rounded) - len(
            shutdown_ramp)
        shutdown_ramp_padded = shutdown_ramp + [shutdown_ramp[-1]] * shutdown_ramp_padding_size
        shutdown_ramp_interpolated = np.reshape(shutdown_ramp_padded, (-1, converted_time_rounded)).mean(axis=1)

        disp_res = power_variables + a.conversion_factor_power_heat * heat_variables
        disp_true = [0] * 5 + list(start_ramp_interpolated) + list(reversed(shutdown_ramp_interpolated)) + [0] * (timegrid.T-len(start_ramp_interpolated)-len(shutdown_ramp_interpolated)-5)
        self.assertAlmostEqual(abs(disp_res - disp_true).sum(), 0, 4)
        start_variables_true = np.zeros(timegrid.T)
        start_variables_true[5] = 1
        self.assertAlmostEqual(abs(start_variables_true - start_variables).sum(), 0, 4)
        shutdown_variables_true = np.zeros(timegrid.T)
        shutdown_variables_true[5 + len(start_ramp_interpolated) + len(shutdown_ramp_interpolated)] = 1
        self.assertAlmostEqual(abs(shutdown_variables_true - shutdown_variables).sum(), 0, 4)

    def test_start_and_shutdown_ramp_heat_1(self):
        """ Testing heat start ramp
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        node_gas = eao.assets.Node('node_gas')

        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 2)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')
        # start ramps
        start_ramp_lower_bounds = [10, 20, 30, 40]
        start_ramp_upper_bounds = [10, 20, 30, 40]
        shutdown_ramp_lower_bounds = [10, 40, 45, 50, 50]
        shutdown_ramp_upper_bounds = [10, 40, 45, 50, 50]

        # heat ... NEW
        start_ramp_lower_bounds_heat = [0, 0, 0, 0]
        start_ramp_upper_bounds_heat = [0, 100, 100, 100]
        shutdown_ramp_lower_bounds_heat = [0, 0, 0, 0, 0]
        shutdown_ramp_upper_bounds_heat = [0, 100, 100, 100, 100]

        a = eao.assets.CHPAsset(name='CHP',
                                nodes=(node_power, node_heat, node_gas),
                                min_cap=1.,
                                max_cap=50.,
                                start_costs=1.,
                                running_costs=5.,
                                conversion_factor_power_heat= 0.2,
                                max_share_heat= 1,
                                start_fuel = 10,
                                fuel_efficiency= .5,
                                consumption_if_on= .1,
                                start_ramp_lower_bounds=start_ramp_lower_bounds,
                                start_ramp_upper_bounds=start_ramp_upper_bounds,
                                shutdown_ramp_lower_bounds=shutdown_ramp_lower_bounds,
                                shutdown_ramp_upper_bounds=shutdown_ramp_upper_bounds,
                                start_ramp_lower_bounds_heat=start_ramp_lower_bounds_heat,
                                start_ramp_upper_bounds_heat=start_ramp_upper_bounds_heat,
                                shutdown_ramp_lower_bounds_heat=shutdown_ramp_lower_bounds_heat,
                                shutdown_ramp_upper_bounds_heat=shutdown_ramp_upper_bounds_heat

                                )
        b = eao.assets.SimpleContract(name = 'powerMarket', price='price', nodes = node_power, min_cap=-100, max_cap=100)
        c = eao.assets.SimpleContract(name = 'gasMarket', price='priceGas', nodes = node_gas, min_cap=-100, max_cap=100)
        d = eao.assets.SimpleContract(name = 'heatMarket', nodes = node_heat, min_cap='heat_demand', max_cap='heat_demand')
        prices ={'price': -50.*np.ones(timegrid.T),
                 'priceGas': 10*np.ones(timegrid.T),
                 'heat_demand': np.zeros(timegrid.T)}
        prices['heat_demand'][10:20] = -1
        portf = eao.portfolio.Portfolio([a, b, c, d])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        # need to start power an hour before heat
        self.assertAlmostEqual(out['dispatch']['CHP (node_power)'][9], 10, 4)

    def test_start_and_shutdown_ramp_heat_2(self):
        """ Testing heat start ramp
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        node_gas = eao.assets.Node('node_gas')

        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 2)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')
        # start ramps
        start_ramp_lower_bounds = [10, 20, 30, 40]
        start_ramp_upper_bounds = [10, 20, 30, 40]
        shutdown_ramp_lower_bounds = [10, 40, 45, 50, 50]
        shutdown_ramp_upper_bounds = [10, 40, 45, 50, 50]

        # heat ... NEW
        start_ramp_lower_bounds_heat = [1, 2, 3, 4]
        start_ramp_upper_bounds_heat = [1, 2, 3, 4]
        shutdown_ramp_lower_bounds_heat = [1, 2, 3, 4, 5]
        shutdown_ramp_upper_bounds_heat = [1, 2, 3, 4, 5]
        cr = .2
        a = eao.assets.CHPAsset(name='CHP',
                                nodes=(node_power, node_heat, node_gas),
                                min_cap=1.,
                                max_cap=50.,
                                start_costs=0.,
                                running_costs=5.,
                                conversion_factor_power_heat= cr,
                                max_share_heat= 1,
                                start_fuel = 10,
                                fuel_efficiency= .5,
                                consumption_if_on= .1,
                                start_ramp_lower_bounds=start_ramp_lower_bounds,
                                start_ramp_upper_bounds=start_ramp_upper_bounds,
                                shutdown_ramp_lower_bounds=shutdown_ramp_lower_bounds,
                                shutdown_ramp_upper_bounds=shutdown_ramp_upper_bounds,
                                start_ramp_lower_bounds_heat=start_ramp_lower_bounds_heat,
                                start_ramp_upper_bounds_heat=start_ramp_upper_bounds_heat,
                                shutdown_ramp_lower_bounds_heat=shutdown_ramp_lower_bounds_heat,
                                shutdown_ramp_upper_bounds_heat=shutdown_ramp_upper_bounds_heat
                                )
        b = eao.assets.SimpleContract(name = 'powerMarket', price='price', nodes = node_power, min_cap=-100, max_cap=100)
        c = eao.assets.SimpleContract(name = 'gasMarket', price='priceGas', nodes = node_gas, min_cap=-500, max_cap=500)
        d = eao.assets.SimpleContract(name = 'heatMarket', nodes = node_heat, min_cap=-100, max_cap=100)
        prices ={'price': 0.*np.ones(timegrid.T),
                 'priceGas': 10*np.ones(timegrid.T)}
#                 'heat_demand': np.zeros(timegrid.T)}
#        prices['heat_demand'][10:20] = -1
        prices['price'][10:20] = 1000
        portf = eao.portfolio.Portfolio([a, b, c, d])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        # need to start power an hour before heat
        myr = out['dispatch']['CHP (node_heat)'][6:10].values
        myrp = out['dispatch']['CHP (node_power)'][6:10].values
        myrt = myr*cr+myrp # total
        # check start ramps are fulfilled
        for i in range(0,4):
             self.assertAlmostEqual(myr[i], start_ramp_lower_bounds_heat[i], 4)
                # check power side ... total virtual dispatch
             self.assertAlmostEqual(myrt[i], start_ramp_lower_bounds[i], 4)

class CHPAssetTest_with_threshhold(unittest.TestCase):

    def test_basics(self):
        """ Unit test. Test inheritance of CHPAsset class
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.CHPAsset_with_min_load_costs(name='CHP', price='rand_price', nodes = (node_power, node_heat),
                                min_cap=5., max_cap=10.)
        prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        x_power = np.around(res.x[:timegrid.T], decimals = 3) # round
        x_heat = np.around(res.x[timegrid.T:2*timegrid.T], decimals = 3) # round

        check =     all(x_power[np.sign(op.c[:timegrid.T]) == -1] + a.conversion_factor_power_heat * x_heat[np.sign(op.c[:timegrid.T]) == -1] == op.u[:timegrid.T][np.sign(op.c[:timegrid.T]) == -1]) \
                and all(x_power[np.sign(op.c[:timegrid.T]) == 1] + a.conversion_factor_power_heat * x_heat[np.sign(op.c[:timegrid.T]) == 1] == 0)
        tot_dcf = np.around((a.dcf(op, res)).sum(), decimals = 3) # asset dcf, calculated independently
        check = check and (tot_dcf == np.around(res.value , decimals = 3))
        self.assertTrue(check)

        # serialization
        a = eao.assets.CHPAsset(name='CHP', price='rand_price', nodes = (node_power, node_heat),
                                min_cap=5., max_cap=10.)
        tt = eao.serialization.to_json(a)
        aa = eao.serialization.load_from_json(tt)
        self.assertAlmostEqual(aa.min_cap, a.min_cap, 5)

        a = eao.assets.CHPAsset_with_min_load_costs(name='CHP', price='rand_price', nodes = (node_power, node_heat),
                                min_cap=5., max_cap=10., min_load_costs=5, min_load_threshhold=1)
        tt = eao.serialization.to_json(a)
        aa = eao.serialization.load_from_json(tt)
        self.assertAlmostEqual(aa.min_cap, a.min_cap, 5)
        self.assertAlmostEqual(aa.min_load_costs, a.min_load_costs, 5)

        a = eao.assets.CHPAsset_with_min_load_costs(name='CHP', price='rand_price', nodes = (node_power, node_heat),
                                min_cap=5., max_cap=10.)
        tt = eao.serialization.to_json(a)
        aa = eao.serialization.load_from_json(tt)
        self.assertAlmostEqual(aa.min_cap, a.min_cap, 5)

    def test_optimization(self):
        """ Unit test. Test simple case with theshhold
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,2), freq = 'h')
        # define a CHP with extra costs when dispatch is below threshhold of 4
        a = eao.assets.CHPAsset_with_min_load_costs(name='CHP', price='price', nodes = (node_power, node_heat),
                                min_cap=2., max_cap=10.,
                                min_load_threshhold= 4,
                                min_load_costs = 1.)
        demand = eao.assets.SimpleContract(name = 'demand',
                                           nodes = node_power,
                                           min_cap = 'demand', max_cap = 'demand')
        prices ={'price':  np.zeros(timegrid.T),
                 'demand': -np.linspace(2, 10, timegrid.T)}
        portf = eao.portfolio.Portfolio([demand, a])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        costs = out['DCF']['CHP'].values
        disp  = out['dispatch']['CHP (node_power)'].values
        # check: costs below threshhold are 1, above 0
        self.assertAlmostEqual(np.abs(costs[disp>=4]).sum(), 0, 5)
        self.assertAlmostEqual(np.abs(costs[disp<4]+1).sum(), 0, 5)

    def test_optimization_vectors(self):
        """ Unit test. Test simple case with theshhold
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,2), freq = 'h')
        # define a CHP with extra costs when dispatch is below threshhold of 4
        a = eao.assets.CHPAsset_with_min_load_costs(name='CHP', price='price', nodes = (node_power, node_heat),
                                min_cap=2., max_cap=10.,
                                min_load_threshhold= 'ml_t',
                                min_load_costs = 'ml_c')
        demand = eao.assets.SimpleContract(name = 'demand',
                                           nodes = node_power,
                                           min_cap = 'demand', max_cap = 'demand')
        prices ={'price':   np.zeros(timegrid.T),
                 'demand': -np.linspace(2, 10, timegrid.T),
                 'ml_c':    np.linspace(0, 10, timegrid.T),
                 'ml_t':    np.linspace(4, 5, timegrid.T)}
        portf = eao.portfolio.Portfolio([demand, a])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        costs = out['DCF']['CHP'].values
        disp  = out['dispatch']['CHP (node_power)'].values
        # check: costs below threshhold are 1, above 0
        self.assertAlmostEqual(np.abs(costs[disp<prices['ml_t']]+prices['ml_c'][disp<=prices['ml_t']]).sum(), 0, 5)
        self.assertAlmostEqual(np.abs(costs[disp>prices['ml_t']]).sum(), 0, 5)

    def test_isrunning(self):
        """ Unit test. Test simple case with threshhold
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        timegrid = eao.assets.Timegrid(dt.date(2020,2,1), dt.date(2020,2,2), freq = '2h')
        # define a CHP with extra costs when dispatch is below threshhold of 4
        a = eao.assets.CHPAsset_with_min_load_costs(name='CHP', price='price', nodes = (node_power, node_heat),
                                min_cap=2., max_cap=10.,
                                min_load_threshhold= 7,
                                min_load_costs = 1.)
        demand = eao.assets.SimpleContract(name = 'demand',
                                           nodes = node_power,
                                           min_cap = 'demand', max_cap = 'demand')
        prices ={'price':  np.zeros(timegrid.T),
                 'demand': -np.linspace(2, 10, timegrid.T)}
        prices['demand'][0:5] = 0
        portf = eao.portfolio.Portfolio([demand, a])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        costs = out['DCF']['CHP'].values
        disp  = out['dispatch']['CHP (node_power)'].values
        # check: costs below threshhold are 1, above 0
        self.assertAlmostEqual(np.abs(costs[disp==0]).sum(), 0, 5)
        self.assertAlmostEqual(np.abs(costs[(disp<7*2)&(disp>0)]+2).sum(), 0, 5)
        self.assertAlmostEqual(np.abs(costs[(disp>7*2)]).sum(), 0, 5)

    def test_check_indexing(self):
        """ Unit test. Test simple case with threshhold
        """
        # portf = eao.serialization.load_from_json(file_name='out_portf.json')
        # tg = eao.serialization.load_from_json(file_name='out_timegrid.json')
        # prices = {'xx': np.ones(tg.T)}
        # op = portf.setup_optim_problem(prices = prices, timegrid = tg)
        node_power = eao.assets.Node('node_power')
        node_gas = eao.assets.Node('node_gas')
        node_heat = eao.assets.Node('node_heat')
        timegrid = eao.assets.Timegrid(dt.date(2020,2,1), dt.date(2020,2,2), freq = 'h')
        # define a CHP with extra costs when dispatch is below threshhold of 4
        a = eao.assets.CHPAsset_with_min_load_costs(name='CHP', price='price', nodes = (node_power, node_heat, node_gas),
                                min_cap=2., max_cap=10.,
                                min_load_threshhold= 7,
                                start_costs=0.,
                                fuel_efficiency=0.9,
                                min_load_costs = 100)
        demand = eao.assets.SimpleContract(name = 'demand',
                                           nodes = node_power,
                                           min_cap = 'demand', max_cap = 'demand')
        gas = eao.assets.SimpleContract(name = 'gas',
                                           nodes = node_gas,
                                           min_cap = -1000, max_cap = 1000)
        prices ={'price':  np.zeros(timegrid.T),
                 'demand': -np.linspace(2, 10, timegrid.T)}
        prices['demand'][0:5] = 0
        portf = eao.portfolio.Portfolio([demand, a, gas])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        costs = out['DCF']['CHP'].values
        disp  = out['dispatch']['CHP (node_power)'].values
        # check: costs below threshhold are 1, above 0
        self.assertAlmostEqual(np.abs(costs[disp==0]).sum(), 0, 5)
        self.assertAlmostEqual(np.abs(costs[(disp<7)&(disp>0)]+100).sum(), 0, 5)
        self.assertAlmostEqual(np.abs(costs[(disp>7)]).sum(), 0, 5)


class CHPAssetTest_no_heat(unittest.TestCase):
    def test_simple_no_heat(self):
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

        ## new: heat node is None
        a = eao.assets.CHPAsset(name='CHP', price='rand_price',
                                nodes = node_power,  # !!!!! heat node not given or None
                                conversion_factor_power_heat = 0,  # use high number to assert I'd see effects if used
                                min_cap=5., max_cap=10.,
                                _no_heat = True)
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        x_power = np.around(res.x[:timegrid.T], decimals = 3) # round
        self.assertTrue(all(x_power==x_power_o))

    def test_gas_consumption_no_heat(self):
        """ Unit test. Setting up a CHPAsset with explicit gas (fuel) consumption
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        node_gas = eao.assets.Node('node_gas')

        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 2)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        ##################################### original test
        #####################################################

        # simple case, no min run time
        a = eao.assets.CHPAsset(name='CHP',
                                nodes=(node_power, node_heat, node_gas),
                                min_cap=1.,
                                max_cap=10.,
                                start_costs=1.,
                                running_costs=5.,
                                conversion_factor_power_heat= 0.2,
                                max_share_heat= 1,
                                start_fuel = 10,
                                fuel_efficiency= .5,
                                consumption_if_on= .1)
        b = eao.assets.SimpleContract(name = 'powerMarket', price='price', nodes = node_power, min_cap=-100, max_cap=100)
        c = eao.assets.SimpleContract(name = 'gasMarket', price='priceGas', nodes = node_gas, min_cap=-100, max_cap=100)
        d = eao.assets.SimpleContract(name = 'heatMarket', price='priceGas', nodes = node_heat, min_cap=0, max_cap=0)
        prices ={'price': 50.*np.ones(timegrid.T), 'priceGas': 0.1*np.ones(timegrid.T)}
        prices['price'][0:5] = -100.
        portf = eao.portfolio.Portfolio([a, b, c, d])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        # check manually checked values
        check = out['dispatch']['CHP (node_power)'].sum()
        self.assertAlmostEqual(check, 190. , 4)
        check = out['dispatch']['CHP (node_heat)'].sum()
        self.assertAlmostEqual(check, 0. , 4)
        check = out['dispatch']['gasMarket (node_gas)'].sum()
        self.assertAlmostEqual(check, 391.9 , 4)
        #############################  test without heat node
        #####################################################
        # simple case, no min run time
        a = eao.assets.CHPAsset(name='CHP',
                                nodes=(node_power, node_gas),
                                _no_heat = True,
                                min_cap=1.,
                                max_cap=10.,
                                start_costs=1.,
                                running_costs=5.,
                                max_share_heat= 1,
                                start_fuel = 10,
                                fuel_efficiency= .5,
                                consumption_if_on= .1)
        b = eao.assets.SimpleContract(name = 'powerMarket', price='price', nodes = node_power, min_cap=-100, max_cap=100)
        c = eao.assets.SimpleContract(name = 'gasMarket', price='priceGas', nodes = node_gas, min_cap=-100, max_cap=100)
        prices ={'price': 50.*np.ones(timegrid.T), 'priceGas': 0.1*np.ones(timegrid.T)}
        prices['price'][0:5] = -100.
        portf = eao.portfolio.Portfolio([a, b, c])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        # check manually checked values
        check = out['dispatch']['CHP (node_power)'].sum()
        self.assertAlmostEqual(check, 190. , 4)
        check = out['dispatch']['gasMarket (node_gas)'].sum()
        self.assertAlmostEqual(check, 391.9 , 4)

    def test_optimization_vectors_no_heat(self):
        """ Unit test. Same test as above, no heat
        """
        node_power = eao.assets.Node('node_power')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,2), freq = 'h')
        # define a CHP with extra costs when dispatch is below threshhold of 4
        a = eao.assets.CHPAsset_with_min_load_costs(name='CHP', price='price', nodes = (node_power),
                                _no_heat = True,
                                min_cap=2., max_cap=10.,
                                min_load_threshhold= 'ml_t',
                                min_load_costs = 'ml_c')
        demand = eao.assets.SimpleContract(name = 'demand',
                                           nodes = node_power,
                                           min_cap = 'demand', max_cap = 'demand')
        prices ={'price':   np.zeros(timegrid.T),
                 'demand': -np.linspace(2, 10, timegrid.T),
                 'ml_c':    np.linspace(0, 10, timegrid.T),
                 'ml_t':    np.linspace(4, 5, timegrid.T)}
        portf = eao.portfolio.Portfolio([demand, a])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        costs = out['DCF']['CHP'].values
        disp  = out['dispatch']['CHP'].values
        # check: costs below threshhold are 1, above 0
        self.assertAlmostEqual(np.abs(costs[disp<prices['ml_t']]+prices['ml_c'][disp<=prices['ml_t']]).sum(), 0, 5)
        self.assertAlmostEqual(np.abs(costs[disp>prices['ml_t']]).sum(), 0, 5)


class Plant(unittest.TestCase):
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
        a = eao.assets.Plant(name='CHP', price='rand_price',
                                nodes = node_power,  # !!!!! heat node not given or None
                                min_cap=5., max_cap=10.)
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        x_power = np.around(res.x[:timegrid.T], decimals = 3) # round
        self.assertTrue(all(x_power==x_power_o))

    def test_gas_consumption_PP(self):
        """ Unit test. Setting up a CHPAsset with explicit gas (fuel) consumption
        """
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        node_gas = eao.assets.Node('node_gas')

        Start = dt.date(2021, 1, 1)
        End = dt.date(2021, 1, 2)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        ##################################### original test
        #####################################################

        # simple case, no min run time
        a = eao.assets.CHPAsset(name='CHP',
                                nodes=(node_power, node_heat, node_gas),
                                min_cap=1.,
                                max_cap=10.,
                                start_costs=1.,
                                running_costs=5.,
                                conversion_factor_power_heat= 0.2,
                                max_share_heat= 1,
                                start_fuel = 10,
                                fuel_efficiency= .5,
                                consumption_if_on= .1)
        b = eao.assets.SimpleContract(name = 'powerMarket', price='price', nodes = node_power, min_cap=-100, max_cap=100)
        c = eao.assets.SimpleContract(name = 'gasMarket', price='priceGas', nodes = node_gas, min_cap=-100, max_cap=100)
        d = eao.assets.SimpleContract(name = 'heatMarket', price='priceGas', nodes = node_heat, min_cap=0, max_cap=0)
        prices ={'price': 50.*np.ones(timegrid.T), 'priceGas': 0.1*np.ones(timegrid.T)}
        prices['price'][0:5] = -100.
        portf = eao.portfolio.Portfolio([a, b, c, d])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        # check manually checked values
        check = out['dispatch']['CHP (node_power)'].sum()
        self.assertAlmostEqual(check, 190. , 4)
        check = out['dispatch']['CHP (node_heat)'].sum()
        self.assertAlmostEqual(check, 0. , 4)
        check = out['dispatch']['gasMarket (node_gas)'].sum()
        self.assertAlmostEqual(check, 391.9 , 4)
        #############################  test without heat node
        #####################################################
        # simple case, no min run time
        a = eao.assets.Plant(name='PP',
                                nodes=(node_power, node_gas),
                                min_cap=1.,
                                max_cap=10.,
                                start_costs=1.,
                                running_costs=5.,
                                start_fuel = 10,
                                fuel_efficiency= .5,
                                consumption_if_on= .1)
        b = eao.assets.SimpleContract(name = 'powerMarket', price='price', nodes = node_power, min_cap=-100, max_cap=100)
        c = eao.assets.SimpleContract(name = 'gasMarket', price='priceGas', nodes = node_gas, min_cap=-100, max_cap=100)
        prices ={'price': 50.*np.ones(timegrid.T), 'priceGas': 0.1*np.ones(timegrid.T)}
        prices['price'][0:5] = -100.
        portf = eao.portfolio.Portfolio([a, b, c])
        op = portf.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, prices)
        # check manually checked values
        check = out['dispatch']['PP (node_power)'].sum()
        self.assertAlmostEqual(check, 190. , 4)
        check = out['dispatch']['gasMarket (node_gas)'].sum()
        self.assertAlmostEqual(check, 391.9 , 4)

        # check serialization (new class...)
        s = eao.serialization.to_json(a)
        aa = eao.serialization.load_from_json(s)

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
        import os
        myfile = os.path.join(os.path.join(os.path.dirname(__file__)),'plant_test_data.csv')
        df = pd.read_csv(myfile)
        df.set_index('date', inplace = True)
        df.index = pd.to_datetime(df.index, format='%m/%d/%y %I:%M %p')
        df = timegrid.prices_to_grid(df)
        # simple case, no min run time
        a = eao.assets.Plant(name='PP',
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


    def test_PP_check_start_ramp_smaller_mincap(self):
        """ Unit test. Predefined data - checking result is same as checked
        """
        node_power = eao.assets.Node('node_power')
        node_gas = eao.assets.Node('node_gas')

        Start = dt.date(2022, 1, 1)
        End = dt.date(2022, 1, 3)
        timegrid = eao.assets.Timegrid(Start, End, freq='15min')

        #############################  test without heat node
        #####################################################
        # load test data
        import os
        myfile = os.path.join(os.path.join(os.path.dirname(__file__)),'plant_test_data.csv')
        df = pd.read_csv(myfile)
        df.set_index('date', inplace = True)
        df = timegrid.prices_to_grid(df)
        # simple case, no min run time
        a = eao.assets.Plant(name='PP',
                                nodes=(node_power, node_gas),
                                min_cap         = 'mincap',
                                max_cap         = 'maxcap',
                                start_costs     = 1.,
                                running_costs   = 'runC',
                                fuel_efficiency = .5,
                                consumption_if_on= .1,
                                start_fuel      = 1,
                                min_downtime    = 2,
                                ramp            = 8,
                                time_already_running=0,
                                time_already_off= 1,
                                start_ramp_upper_bounds=[1.1],
                                start_ramp_lower_bounds=[1.1],
                                shutdown_ramp_upper_bounds=[2.2],
                                shutdown_ramp_lower_bounds=[2.2],
                                ramp_freq='15min')
        b = eao.assets.SimpleContract(name = 'powerMarket', price='power_price', nodes = node_power, min_cap=-100, max_cap=100)
        c = eao.assets.SimpleContract(name = 'gasMarket', price='gas_price', nodes = node_gas, min_cap=-100, max_cap=100)
        portf = eao.portfolio.Portfolio([a, b, c])
        op = portf.setup_optim_problem(df, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, df)
        ### for checks: eao.io.output_to_file(out, 'results_plant.xlsx')
        # # check manually checked values
        self.assertAlmostEqual(res.value,  34752.9612, 2)
        self.assertAlmostEqual(out['DCF'].sum().sum(),  34752.9612, 2)
        # self.assertAlmostEqual(out['dispatch'].sum().sum(),  0, 2)

        # check serialization (new class...)
        s = eao.serialization.to_json(a)
        aa = eao.serialization.load_from_json(s)



    def test_PP_check_start_ramp_vs_ramp(self):
        """ What happens with ramp smaller / not equal start ramp? should be ignored
        """
        node_power = eao.assets.Node('node_power')
        node_gas = eao.assets.Node('node_gas')

        Start = dt.date(2022, 1, 1)
        End = dt.date(2022, 1, 2)
        timegrid = eao.assets.Timegrid(Start, End, freq='h')

        #############################  test without heat node
        #####################################################
        # load test data
        import os
        myfile = os.path.join(os.path.join(os.path.dirname(__file__)),'plant_test_data.csv')
        df = pd.read_csv(myfile)
        df.set_index('date', inplace = True)
        df = timegrid.prices_to_grid(df)
        df['power_price'] = 1000
        df['mincap'] = 6
        # simple case, no min run time
        a = eao.assets.Plant(name='PP',
                                nodes=(node_power, node_gas),
                                min_cap         = 'mincap',
                                max_cap         = 'maxcap',
                                start_costs     = 1.,
                                running_costs   = 'runC',
                                fuel_efficiency = 1,
                                consumption_if_on= .1,
                                start_fuel      = 1,
                                min_downtime    = 2,
                                ramp            = 1.1,
                                time_already_running=0,
                                time_already_off= 1,
                                start_ramp_upper_bounds=[1,2,4,6,10],
                                start_ramp_lower_bounds=[1,2,4,6,10],
                                shutdown_ramp_upper_bounds=[1.1],
                                shutdown_ramp_lower_bounds=[1.1],
                                ramp_freq='h')
        b = eao.assets.SimpleContract(name = 'powerMarket', price='power_price', nodes = node_power, min_cap=-100, max_cap=100)
        c = eao.assets.SimpleContract(name = 'gasMarket', price='gas_price', nodes = node_gas, min_cap=-100, max_cap=100)
        portf = eao.portfolio.Portfolio([a, b, c])
        op = portf.setup_optim_problem(df, timegrid=timegrid)
        res = op.optimize()
        out = eao.io.extract_output(portf, op, res, df)
        # dispatch should follow start ramp and then add ramp
        self.assertAlmostEqual(out['dispatch'].iloc[0,0],  0, 4)
        self.assertAlmostEqual(out['dispatch'].iloc[1,0],  1, 4)
        self.assertAlmostEqual(out['dispatch'].iloc[2,0],  2, 4)
        self.assertAlmostEqual(out['dispatch'].iloc[3,0],  4, 4)
        self.assertAlmostEqual(out['dispatch'].iloc[4,0],  6, 4)
        self.assertAlmostEqual(out['dispatch'].iloc[5,0],  10, 4)
        self.assertAlmostEqual(out['dispatch'].iloc[6,0],  11.1, 4)
        self.assertAlmostEqual(out['dispatch'].iloc[7,0],  12.2, 4)


class CHPAssetTest_with_PQ_polygon(unittest.TestCase):

    def test_check_polygon(self):
        """ Unit test. PQ diagram given as polygon. Test checker /convex/ orientation """
        # most basic asset
        node_power = eao.assets.Node('node_power')
        node_heat = eao.assets.Node('node_heat')
        timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
        a = eao.assets.CHP_PQ_diagram(name='CHP', price='rand_price', nodes = (node_power, node_heat),
                                min_cap=5., max_cap=10.)

        poly = [[0,0], [0, 100], [100, 100], [100, 0]] # sqare, clockwise
        self.assertEqual(a._check_polygon(poly), 1)
        poly = [[50.,50.], [0., 0.], [50., 0.]] # triangle, counter clockwise
        self.assertEqual(a._check_polygon(poly), -1)
        poly = [[0,0], [0, 100], [50, 50], [100, 100], [100, 0]] # sqare with dip - non convex
        self.assertEqual(a._check_polygon(poly), 0)
        poly = [[50,50], [0, 0], [25, 0], [50, 0]] # two points on line - should be ok
        poly = [np.array(p) for p in poly] # also test with arrays
        self.assertEqual(a._check_polygon(poly), -1)

    def test_build_polygon_asset_square(self):
        """ Unit test. PQ diagram given as polygon. SQUARE """
        # most basic asset
        node_power = eao.assets.Node('node_power')
        node_heat  = eao.assets.Node('node_heat')
        timegrid   = eao.assets.Timegrid(dt.date(2024,1,1), dt.date(2024,1,2), freq = 'h')
        data = {'price': np.sin(np.linspace(0,20*np.pi, timegrid.T)),
                'heat' : -50*(np.linspace(0,1, timegrid.T))-15}
        # square
        #        [P, Q]
        poly = [[ 5, 15],
                [10, 15],
                [10, 25],
                [ 5, 25]]
        # no further restriction - instantaneous reaction to prices
        m = eao.assets.SimpleContract(name = 'market', price='price', nodes = node_power, min_cap=-1000, max_cap=1000)
        h = eao.assets.SimpleContract(name = 'heat', nodes = node_heat, min_cap='heat', max_cap='heat')
        a = eao.assets.CHP_PQ_diagram(name='poly',
                                      nodes = (node_power, node_heat),
                                      min_cap=0.,
                                      max_cap=500.,   # generous, restr only from poly!
                                      conversion_factor_power_heat=0.2,
                                      pq_polygon =poly)
        portf = eao.portfolio.Portfolio([m, a, h])
        op = portf.setup_optim_problem(prices=data, timegrid=timegrid)
        #### check limits for this setup
        # power limits
        ind = op.mapping[(op.mapping['asset']=='poly')&(op.mapping['node']=='node_power')&(op.mapping['type']=='d')].index.values
        self.assertTrue(all(op.l[ind]==5))
        self.assertTrue(all(op.u[ind]==10))
        # heat limits
        ind = op.mapping[(op.mapping['asset']=='poly')&(op.mapping['node']=='node_heat')&(op.mapping['type']=='d')].index.values
        self.assertTrue(all(op.l[ind]==15))
        self.assertTrue(all(op.u[ind]==25))
        ##### optimize
        out   = eao.optimize(portf, timegrid, data)
        self.assertTrue(out['summary']['status'] == 'not successful') # heat delivery cannot be met with given poly

    def test_build_polygon_asset_tetra(self):
        """ Unit test. PQ diagram given as polygon. tetraeder """
        # most basic asset
        node_power = eao.assets.Node('node_power')
        node_heat  = eao.assets.Node('node_heat')
        timegrid   = eao.assets.Timegrid(dt.date(2024,1,1), dt.date(2024,1,20), freq = 'h')
        data = {'price': np.sin(np.linspace(0,20*np.pi, timegrid.T)),
                'heat' : -5-25*(np.linspace(0,1, timegrid.T))}
        # square
        #        [P, Q]
        poly = [[ 5, 5],
                [10, 15],
                [ 8, 25],
                [ 6, 30]]
        poly_eps = [[ 4.99, 4.88], # making it a bit bigger to avoid numerical issues
                    [10.01, 14.99],
                    [ 8.01, 25.01],
                    [ 5.99, 30.03]]        
        # no further restriction - instantaneous reaction to prices
        m = eao.assets.SimpleContract(name = 'market', price='price', nodes = node_power, min_cap=-1000, max_cap=1000)
        h = eao.assets.SimpleContract(name = 'heat', nodes = node_heat, min_cap='heat', max_cap='heat')
        a = eao.assets.CHP_PQ_diagram(name='poly',
                                      nodes = (node_power, node_heat),
                                      min_cap=0.,
                                      max_cap=500.,   # generous, restr only from poly!
                                      conversion_factor_power_heat=0.2,
                                      pq_polygon =poly)
        portf = eao.portfolio.Portfolio([m, a, h])
        ##### optimize
        out   = eao.optimize(portf, timegrid, data)
        x = out['dispatch']["poly (node_heat)"].values
        y = out['dispatch']["poly (node_power)"].values
        ### look at dispatch along edges
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.scatter(x, y)
        # fig.show()
        #### test: any dispatch outside polygon?

        import matplotlib.path as mplPath
        poly_path = mplPath.Path(poly_eps)
        for i in range(len(x)):
            point = np.array([y[i], x[i]])
            # print(point, " is in polygon: ", poly_path.contains_point(point))
            self.assertTrue(poly_path.contains_point(point), msg=f"point {point} not in polygon")
        return


    def test_build_polygon_asset_mixed_straight_tetra(self):
        """ Unit test. PQ diagram given as polygon. tetraeder """
        # most basic asset
        node_power = eao.assets.Node('node_power')
        node_heat  = eao.assets.Node('node_heat')
        timegrid   = eao.assets.Timegrid(dt.date(2024,1,1), dt.date(2024,1,20), freq = 'h')
        data = {'price': np.sin(np.linspace(0,20*np.pi, timegrid.T)),
                'heat' : -5-25*(np.linspace(0,1, timegrid.T))}
        # square
        #        [P, Q]
        poly = [[ 5, 5],
                [10, 5],
                [ 8, 25],
                [ 6, 30]]
        poly_eps = [[ 4.99, 4.99], # making it a bit bigger to avoid numerical issues
                    [10.01, 4.99],
                    [ 8.01, 25.01],
                    [ 5.99, 30.03]]        
        # no further restriction - instantaneous reaction to prices
        m = eao.assets.SimpleContract(name = 'market', price='price', nodes = node_power, min_cap=-1000, max_cap=1000)
        h = eao.assets.SimpleContract(name = 'heat', nodes = node_heat, min_cap='heat', max_cap='heat')
        a = eao.assets.CHP_PQ_diagram(name='poly',
                                      nodes = (node_power, node_heat),
                                      min_cap=0.,
                                      max_cap=500.,   # generous, restr only from poly!
                                      conversion_factor_power_heat=0.2,
                                      pq_polygon =poly)
        portf = eao.portfolio.Portfolio([m, a, h])
        ##### optimize
        out   = eao.optimize(portf, timegrid, data)
        x = out['dispatch']["poly (node_heat)"].values
        y = out['dispatch']["poly (node_power)"].values
        #### test: any dispatch outside polygon?
        import matplotlib.path as mplPath
        poly_path = mplPath.Path(poly_eps)
        for i in range(len(x)):
            point = np.array([y[i], x[i]])
            # print(point, " is in polygon: ", poly_path.contains_point(point))
            self.assertTrue(poly_path.contains_point(point), msg=f"point {point} not in polygon")
        return

    def test_polygon_serialization(self):
        """ Unit test. PQ diagram given as polygon. tetraeder """
        # most basic asset
        node_power = eao.assets.Node('node_power')
        node_heat  = eao.assets.Node('node_heat')
        timegrid   = eao.assets.Timegrid(dt.date(2024,1,1), dt.date(2024,1,20), freq = 'h')
        poly = [[ 5, 5],
                [10, 15],
                [ 8, 25],
                [ 6, 30]]
        # no further restriction - instantaneous reaction to prices
        m = eao.assets.SimpleContract(name = 'market', price='price', nodes = node_power, min_cap=-1000, max_cap=1000)
        h = eao.assets.SimpleContract(name = 'heat', nodes = node_heat, min_cap='heat', max_cap='heat')
        a = eao.assets.CHP_PQ_diagram(name='poly',
                                      nodes = (node_power, node_heat),
                                      min_cap=0.,
                                      max_cap=500.,   # generous, restr only from poly!
                                      conversion_factor_power_heat=0.2,
                                      pq_polygon = poly)
        portf = eao.portfolio.Portfolio([m, a, h])
        s = eao.serialization.to_json(a)
        aa = eao.serialization.load_from_json(s)
        self.assertEqual(a.pq_polygon, aa.pq_polygon)
        s = eao.serialization.to_json(portf)
        pp = eao.serialization.load_from_json(s)
        self.assertEqual(a.pq_polygon, pp.assets[1].pq_polygon)


###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
