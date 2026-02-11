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
    def renewable_and_contract(self, timegrid, prices, profile, subsidy=None, fixed_price=None, n_hour_rule=None):
        node = eao.assets.Node("testNode")
        a = eao.assets.RenewableAsset(
            "RenewableAsset",
            node,
            price="price",
            profile=profile,
            subsidy=subsidy,
            fixed_price=fixed_price,
            n_hour_rule=n_hour_rule,
        )
        c = eao.assets.Contract(
            name="Contract",
            price="price",
            nodes=node,
            min_cap=-100.0,
            max_cap=0,
        )
        portf = eao.portfolio.Portfolio([a, c])
        return eao.optimize(portf, timegrid, prices)

    def test_renewables_profile(self):

        start = dt.date(2021, 1, 1)
        end = dt.date(2021, 1, 10)
        timegrid = eao.assets.Timegrid(start, end, freq="h")

        # dummy profile for a PV plant over (end-start) sunny days ~ cos^2
        restr_times = pd.date_range(start, end, freq="h", inclusive="left")
        profile = {}
        profile["start"] = restr_times.to_list()
        profile["end"] = (restr_times + dt.timedelta(hours=1)).to_list()
        profile["values"] = 10 * np.cos(np.linspace(0, 24, len(profile["start"]))) ** 2

        # Constant positive prices, expectation: dispatch = maximal power = energy profile
        prices = {"price":  np.ones(timegrid.T) * 5}
        out_subsidy = self.renewable_and_contract(timegrid, prices, profile, subsidy=2)
        out_fixed_price = self.renewable_and_contract(timegrid, prices, profile, fixed_price=7)
        np.testing.assert_almost_equal(out_subsidy["dispatch"]["RenewableAsset"].values, profile['values'], 4)
        np.testing.assert_almost_equal(out_fixed_price["dispatch"]["RenewableAsset"].values, profile['values'], 4)

        # 5h long period of negative prices, expectation: no dispatch during this period, else dispatch = profile
        prices["price"][5:10] = -5
        out_subsidy = self.renewable_and_contract(timegrid, prices, profile, subsidy=2, n_hour_rule=5)
        out_fixed_price = self.renewable_and_contract(timegrid, prices, profile, fixed_price=7, n_hour_rule=5)
        np.testing.assert_almost_equal(out_subsidy["dispatch"]["RenewableAsset"].values[:5], profile['values'][:5], 4)
        np.testing.assert_almost_equal(out_subsidy["dispatch"]["RenewableAsset"].values[5:10], 0, 4)
        np.testing.assert_almost_equal(out_subsidy["dispatch"]["RenewableAsset"].values[10:], profile['values'][10:], 4)
        np.testing.assert_almost_equal(out_fixed_price["dispatch"]["RenewableAsset"].values[:5], profile['values'][:5], 4)
        np.testing.assert_almost_equal(out_fixed_price["dispatch"]["RenewableAsset"].values[5:10], 0, 4)
        np.testing.assert_almost_equal(out_fixed_price["dispatch"]["RenewableAsset"].values[10:], profile['values'][10:], 4)


if __name__ == '__main__':
    unittest.main()
