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
    def test_renewables(self):
        start = dt.date(2021, 1, 1)
        end = dt.date(2021, 1, 10)
        timegrid = eao.assets.Timegrid(start, end, freq="h")

        # dummy profile for a PV plant over (end-start) sunny days ~ cos^2
        restr_times = pd.date_range(start, end, freq="h", inclusive="left")
        profile = {}
        profile["start"] = restr_times.to_list()
        profile["end"] = (restr_times + dt.timedelta(hours=1)).to_list()
        profile["values"] = 10 * np.cos(np.linspace(0, 24, len(profile["start"]))) ** 2

        node = eao.assets.Node("testNode")
        a = eao.assets.RenewableAsset(
            "RenewableAsset",
            node,
            min_cap=0,
            max_cap=10,
            profile=profile,
            subsidy=2,
            fixed_price=None,
        )
        c = eao.assets.Contract(
            name="c2",
            price="price",
            nodes=node,
            min_cap=-10.0,
            max_cap=0,
        )
        prices = {"price": np.linspace(-100, 100, timegrid.T)}
        #prices = {"price":  np.ones(timegrid.T) * 5}
        portf = eao.portfolio.Portfolio([a, c])
        out = eao.optimize(portf, timegrid, prices)
        print(out)

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
