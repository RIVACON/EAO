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
        # dummy profile 14h almost off, 10h on
        pattern = np.array([0.1] * 14 + [10] * 10)
        restr_times = pd.date_range(start, end, freq="h", inclusive="left")
        self.profile: eao.StartEndValueDict = {
            "start": restr_times.to_list(),
            "end": (restr_times + dt.timedelta(hours=1)).to_list(),
            "values": np.tile(pattern, int(np.ceil(T / len(pattern))))[:T]
        }
        # const. market prices
        self.market_price: eao.StartEndValueDict = {
            "start": restr_times.to_list(),
            "end": (restr_times + dt.timedelta(hours=1)).to_list(),
            "values": np.ones(T) * 5
        }
        self.node = eao.assets.Node("testNode")
        self.renewable = eao.assets.RenewableAsset(
            "Renewable",
            self.node,
            profile=self.profile,
            market_price=self.market_price,
            fixed_price=0,
            n_hour_rule_payment=None,
            n_hour_rule_delivery=None,
            controllable=True,
            sell_position=False,
            cfd_type=False
        )
        self.contract = eao.assets.Contract(
            name="Contract",
            price="price",
            nodes=self.node,
            min_cap=-100.0,
            max_cap=0,
        )
        self.portf = eao.portfolio.Portfolio([self.renewable, self.contract])

    def test_default(self):
        # Constant positive prices, expectation: dispatch = maximal power = energy profile independent of fixed_prices
        # as long as fixed_price < market prices = 5
        for fixed_price in (0, 1, 4):
            self.renewable.fixed_price=fixed_price
            out = eao.optimize(self.portf, self.timegrid, {"price": self.market_price["values"]})
            np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values, self.profile['values'], 4)

    def test_cfd_type(self):
        # Constant positive prices, expectation: dispatch = maximal power = energy profile independent of fixed_prices
        self.renewable.cfd_type = True
        for fixed_price in (0, 1, 4, 6, 9):
            self.renewable.fixed_price=fixed_price
            out = eao.optimize(self.portf, self.timegrid, {"price": self.market_price["values"]})
            np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values, self.profile['values'], 4)

    def test_negative_prices_not_controllable(self):
        # Constant positive prices, except for 5h period. Expectation: No change of dispatch if asset is NOT controllable
        market_price = self.market_price.copy()
        market_price["values"][12:17] = -1
        self.renewable.market_price=market_price
        self.renewable.controllable = False
        out = eao.optimize(self.portf, self.timegrid, {"price": market_price["values"]})
        np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values, self.profile['values'], 4)

    def test_negative_prices_controllable(self):
        # Constant positive prices, except for 5h period. Expectation: No dispatch during neg. prices
        market_price = self.market_price.copy()
        market_price["values"][12:17] = -1
        self.renewable.market_price=market_price
        out = eao.optimize(self.portf, self.timegrid, {"price": market_price["values"]})
        np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values[:12], self.profile['values'][:12], 4)
        np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values[12:17], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values[17:], self.profile['values'][17:], 4)

    def test_n_hour_delivery(self):
        # Constant positive prices, except for 5h period. Expectation: No dispatch during neg. prices
        self.renewable.n_hour_delivery = 5
        market_price = self.market_price.copy()
        market_price["values"][12:17] = -1
        self.renewable.market_price=market_price
        out = eao.optimize(self.portf, self.timegrid, {"price": market_price["values"]})
        np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values[:12], self.profile['values'][:12], 4)
        np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values[12:17], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values[17:], self.profile['values'][17:], 4)

    def test_n_hour_payment(self):
        # Constant positive prices, except for 5h period. Expectation: No dispatch during neg. prices
        self.renewable.n_hour_payment = 5
        market_price = self.market_price.copy()
        market_price["values"][12:17] = -1
        self.renewable.market_price=market_price
        out = eao.optimize(self.portf, self.timegrid, {"price": market_price["values"]})
        np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values[:12], self.profile['values'][:12], 4)
        np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values[12:17], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["Renewable"].values[17:], self.profile['values'][17:], 4)


if __name__ == '__main__':
    unittest.main()
