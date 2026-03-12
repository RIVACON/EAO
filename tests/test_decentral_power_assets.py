import unittest
import numpy as np
import pandas as pd
import datetime as dt
import json
from os.path import dirname, join
import sys

mypath = dirname(__file__)
sys.path.append(join(mypath, ".."))

import eaopack as eao


class RenewablesTestCase(unittest.TestCase):
    def setUp(self):
        start = dt.date(2021, 1, 1)
        end = dt.date(2021, 1, 10)
        self.timegrid = eao.assets.Timegrid(start, end, freq="h")
        T = self.timegrid.T
        self.profile = np.linspace(0, 50, T) + 5.0
        # const. market prices
        self.market_price = np.ones(T) * 5.0
        self.node = eao.assets.Node("testNode")
        self.renewable = eao.assets.RenewableAsset(
            "Renewable",
            self.node,
            profile=self.profile,
            market_price=self.market_price,
            fixed_price=0.0,
            n_hour_rule_payment=None,
            n_hour_rule_delivery=None,
            controllable=True,
            short_position=False,
            cfd_type=False,
        )
        self.contract = eao.assets.Contract(
            name="Contract",
            price=self.market_price,
            nodes=self.node,
            min_cap=-100.0,
            max_cap=100.0,
        )
        self.portf = eao.portfolio.Portfolio([self.renewable, self.contract])

    def test_default(self):
        # Constant positive prices, expectation: dispatch = maximal power = energy profile independent of fixed_prices
        # as long as -fixed_price < market prices = 5
        for fixed_price in (0, -1.0, -4.0):
            self.renewable.fixed_price = fixed_price
            out = eao.optimize(self.portf, self.timegrid)
            np.testing.assert_almost_equal(
                out["dispatch"]["Renewable"].values, self.renewable.profile, 4
            )
        # fixed price > market price, expectation: dispatch everywhere = 0:
        self.renewable.fixed_price = 6
        out = eao.optimize(self.portf, self.timegrid)
        np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values, 0.0, 4)

    def test_cfd_type(self):
        """Test CDF Type of fixed subsidy payments"""
        # CDF Type
        self.renewable.cfd_type = True
        for contr in [True, False]:
            self.renewable.set_timegrid(self.timegrid)
            self.renewable.controllable = contr
            for fixed_price in (-0.1, -1, -9):  # neg: we get
                fp = self.renewable.make_vector(fixed_price)
                fp[1:15] = 10.0  #  effectively we pay
                self.renewable.fixed_price = fp
                out = eao.optimize(self.portf, self.timegrid)
                disp = out["dispatch"]["Renewable"].values
                dcf = out["DCF"]["Renewable"].values
                # cfd logic: we get fixed + market
                we_get = -disp * (fp + self.market_price)
                if not contr:
                    np.testing.assert_almost_equal(disp, self.profile, 4)
                else:
                    myp = np.where(fp > 0.0, 0.0, self.profile)
                    np.testing.assert_almost_equal(disp, myp, 4)
                np.testing.assert_almost_equal(dcf, we_get, 4)
        # CDF Type FALSE
        self.renewable.cfd_type = False
        for contr in [True, False]:
            self.renewable.set_timegrid(self.timegrid)
            self.renewable.controllable = contr
            mp = self.market_price
            mp[0:5] = -20.0
            mp[5:10] = 20.0
            for fixed_price in (-0.1, -9.0):
                fp = fixed_price  # self.renewable.make_vector(fixed_price)
                self.renewable.fixed_price = fp
                out = eao.optimize(self.portf, self.timegrid, {"price": mp})
                disp = out["dispatch"]["Renewable"].values
                dcf = out["DCF"]["Renewable"].values
                # NO cfd logic: we get fixed
                we_get = -disp * (fp)
                if not contr:
                    np.testing.assert_almost_equal(disp, self.profile, 4)
                else:
                    myp = np.where(-fp + self.market_price < 0.0, 0.0, self.profile)
                    np.testing.assert_almost_equal(disp, myp, 4)
                np.testing.assert_almost_equal(dcf, we_get, 4)

    def test_n_hour_delivery(self):
        # Constant positive prices, except for 5h period. Expectation: No dispatch during neg. prices
        self.renewable.n_hour_rule_delivery = 5
        mp = self.market_price.copy()
        mp[12:17] = -1
        self.renewable.controllable = False
        self.renewable.market_price = mp
        out = eao.optimize(self.portf, self.timegrid)
        dp = out["dispatch"]["Renewable"].values
        np.testing.assert_almost_equal(dp[:12], self.profile[:12], 4)
        np.testing.assert_almost_equal(dp[12:17], 0, 4)
        np.testing.assert_almost_equal(dp[17:], self.profile[17:], 4)
        # same as short position
        self.renewable.short_position = True
        out = eao.optimize(self.portf, self.timegrid)
        dp = out["dispatch"]["Renewable"].values
        np.testing.assert_almost_equal(dp[:12], -self.profile[:12], 4)
        np.testing.assert_almost_equal(dp[12:17], 0.0, 4)
        np.testing.assert_almost_equal(dp[17:], -self.profile[17:], 4)

    def test_n_hour_payment(self):
        # Constant positive prices, except for 5h period. Expectation: No dispatch during neg. prices
        self.renewable.n_hour_rule_payment = 4
        mp = self.market_price.copy()
        mp[12:22] = -1.0  # more
        mp[24:28] = -1.0  # exact
        mp[0:3] = -1.0  # less
        fp = -2.0
        self.renewable.fixed_price = fp
        self.renewable.market_price = mp
        ## intermediate: check serialization
        s = eao.serialization.to_json(self.portf)
        portf = eao.serialization.load_from_json(s)
        np.testing.assert_almost_equal(
            portf.get_asset("Renewable").profile, self.profile, 4
        )
        out = eao.optimize(portf, self.timegrid)
        disp = out["dispatch"]["Renewable"].values
        dcf = out["DCF"]["Renewable"].values
        we_get = -disp * fp
        we_get[12:22] = 0.0
        we_get[24:28] = 0.0
        np.testing.assert_almost_equal(dcf, we_get, 4)
        np.testing.assert_almost_equal(
            out["dispatch"]["Renewable"].values, self.profile, 4
        )


class BatteryAsset(unittest.TestCase):

    def test_battery_asset(self):
        """Test specific battery asset - a simplified storage"""
        """trivial test with eff_out"""
        node = eao.assets.Node("testNode")
        timegrid = eao.assets.Timegrid(
            dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq="h"
        )
        a = eao.assets.Battery(
            "Battery",
            node,
            size=5,
            cap_in=1,
            cap_out=1,
            start_level=0,
            end_level=0,
            price="price",
            eff_in=0.8,
            eff_out=0.9,
            no_simult_in_out=True,
        )
        price = np.ones([timegrid.T])
        price[:10] = 0
        price[8] = 5
        price[3:5] = 0
        price[18:20] = 20
        s = eao.serialization.to_json(a)
        a = eao.serialization.from_json(s)
        prices = {"price": price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        xin = res.x[0:24]
        xout = res.x[24:48]
        fl = a.fill_level(op, res)
        self.assertAlmostEqual(
            -xin.sum() / xout.sum(), 1 / 0.9 / 0.8, 3
        )  # overall loss
        self.assertAlmostEqual(fl.max(), 5, 5)
        print(res)
