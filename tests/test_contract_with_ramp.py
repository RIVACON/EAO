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


class Test_Contract_Ramp(unittest.TestCase):

    def simple_case_with_ramp(self, timegrid, prices, ramp1, ramp2) -> dict:
        """Simple case: Two contracts, set up so that we will see no dispatch ... then max"""
        ### manual benchmark
        node = eao.Node("power_node")
        c1 = eao.assets.Contract(
            name="c1", price="price_1", nodes=node, min_cap=0, max_cap=10.0, ramp=ramp1
        )
        c2 = eao.assets.Contract(
            name="c2", price="price_2", nodes=node, min_cap=-10.0, max_cap=0, ramp=ramp2
        )
        # Idea: Give c2 a ramp later on to see the effect
        # a3.set_timegrid(timegrid)
        portf = eao.portfolio.Portfolio([c1, c2])
        out = eao.optimize(portf, timegrid, prices)
        return out

    def test_simple_case_without_ramp(self):
        timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq='15min')
        prices = {
            "price_1": np.ones(timegrid.T) * 2,
            "price_2": -100 * np.ones(timegrid.T),
        }
        prices["price_2"][5:10] = 5  ## here, we will see dispatch c1 ---> c2
        out = self.simple_case_with_ramp(timegrid, prices,None, None)
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[5:10], 2.5, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[5:10], -2.5, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[0:5], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[10:], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[0:5], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[10:], 0, 4)

    def test_simple_case_with_ramp(self):
        for freq in ('15min', '30min', '1h'):
            timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq=freq)
            prices = {
                "price_1": np.ones(timegrid.T) * 2,
                "price_2": -100 * np.ones(timegrid.T),
            }
            prices["price_2"][5:10] = 5  ## here, we will see dispatch c1 ---> c2
            for ramp1 in (0.5, 1, 2):
                for ramp2 in (None, 0.75, 1.25):
                    out = self.simple_case_with_ramp(timegrid, prices, ramp1, ramp2)

                    ramp = ramp1 if ramp2 is None else min(ramp1, ramp2)
                    r = float(pd.Timedelta(freq) / pd.Timedelta('1h'))
                    expected = np.array([r*ramp, 2*r*ramp ,3*r*ramp, 2*r*ramp , r*ramp])
                    np.testing.assert_almost_equal(out["dispatch"]["c1"].values[5:10], expected, 4)
                    expected *= -1
                    np.testing.assert_almost_equal(out["dispatch"]["c2"].values[5:10], expected, 4)

                    np.testing.assert_almost_equal(out["dispatch"]["c1"].values[0:5], 0, 4)
                    np.testing.assert_almost_equal(out["dispatch"]["c1"].values[10:], 0, 4)
                    np.testing.assert_almost_equal(out["dispatch"]["c2"].values[0:5], 0, 4)
                    np.testing.assert_almost_equal(out["dispatch"]["c2"].values[10:], 0, 4)

    def test_simple_case_no_time_offset(self):
        timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq='15min')
        prices = {
            "price_1": np.ones(timegrid.T) * 2,
            "price_2": -100 * np.ones(timegrid.T),
        }
        prices["price_2"][0:5] = 5  ## here, we will see dispatch c1 ---> c2
        out = self.simple_case_with_ramp(timegrid, prices,0.5, None)
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[0:5], np.array([0.125, 0.250, 0.375, 0.250, 0.125]), 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[0:5], np.array([-0.125, -0.250, -0.375, -0.250, -0.125]), 4)
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[5:], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[5:], 0, 4)

    def test_simple_case_no_time_trailing(self):
        timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq='15min')
        T = timegrid.T
        prices = {
            "price_1": np.ones(T) * 2,
            "price_2": -100 * np.ones(T),
        }
        prices["price_2"][T-5:T] = 5  ## here, we will see dispatch c1 ---> c2
        out = self.simple_case_with_ramp(timegrid, prices,0.5, None)
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[T-5:T], np.array([0.125, 0.250, 0.375, 0.500, 0.625]), 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[T-5:T], np.array([-0.125, -0.250, -0.375, -0.5, -0.625]), 4)
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[0:T-5], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[0:T-5], 0, 4)


###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == "__main__":
    unittest.main()
