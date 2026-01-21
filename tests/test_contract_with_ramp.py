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

    def test_simple_case_without_ramp(self):
        """Simple case: Two contracts, set up so that we will see no dispatch ... then max"""
        ### manual benchmark
        node = eao.Node("power_node")
        timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq="15min")
        c1 = eao.assets.SimpleContract(
            name="c1", price="price_1", nodes=node, min_cap=0, max_cap=10.0
        )
        c2 = eao.assets.SimpleContract(
            name="c2", price="price_2", nodes=node, min_cap=-10.0, max_cap=0
        )
        # Idea: Give c2 a ramp later on to see the effect
        # a3.set_timegrid(timegrid)
        prices = {
            "price_1": np.ones(timegrid.T) * 2,
            "price_2": -100 * np.ones(timegrid.T),
        }
        prices["price_2"][5:10] = 5  ## here, we will see dispatch c1 ---> c2
        portf = eao.portfolio.Portfolio([c1, c2])
        out = eao.optimize(portf, timegrid, prices)
        # have a look at out['dispatch'] --> contains the dispatch, i.e. MWh per interval by each contract. Sum must be zero
        # you see that c1 delivers power to c2 in the timesteps 5 to 9
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[5:10], 2.5, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[5:10], -2.5, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[0:5], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[10:], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[0:5], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[10:], 0, 4)

        ## now with ramp: assume, the ramp restricts the change to .5 MWh in 15min
        ## then I expect the dispatch [5] to be .5 instead --> depending on the price levels, the contract may also instead start ramping up earlie
        ## same for dispatch[10] --> now only -5 instead of -2.5
        pass


###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == "__main__":
    unittest.main()
