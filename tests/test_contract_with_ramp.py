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
        timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq="15min")
        prices = {
            "price_1": np.ones(timegrid.T) * 2,
            "price_2": -100 * np.ones(timegrid.T),
        }
        prices["price_2"][5:10] = 5  ## here, we will see dispatch c1 ---> c2
        out = self.simple_case_with_ramp(timegrid, prices, None, None)
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[5:10], 2.5, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[5:10], -2.5, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[0:5], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[10:], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[0:5], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[10:], 0, 4)

    def test_simple_case_with_ramp(self):
        for freq in ("15min", "30min", "1h"):
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
                    r = float(pd.Timedelta(freq) / pd.Timedelta("1h"))
                    expected = np.array(
                        [r * ramp, 2 * r * ramp, 3 * r * ramp, 2 * r * ramp, r * ramp]
                    )
                    np.testing.assert_almost_equal(
                        out["dispatch"]["c1"].values[5:10], expected, 4
                    )
                    expected *= -1
                    np.testing.assert_almost_equal(
                        out["dispatch"]["c2"].values[5:10], expected, 4
                    )

                    np.testing.assert_almost_equal(
                        out["dispatch"]["c1"].values[0:5], 0, 4
                    )
                    np.testing.assert_almost_equal(
                        out["dispatch"]["c1"].values[10:], 0, 4
                    )
                    np.testing.assert_almost_equal(
                        out["dispatch"]["c2"].values[0:5], 0, 4
                    )
                    np.testing.assert_almost_equal(
                        out["dispatch"]["c2"].values[10:], 0, 4
                    )

    def test_simple_case_no_time_offset(self):
        timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq="15min")
        prices = {
            "price_1": np.ones(timegrid.T) * 2,
            "price_2": -100 * np.ones(timegrid.T),
        }
        prices["price_2"][0:5] = 5  ## here, we will see dispatch c1 ---> c2
        out = self.simple_case_with_ramp(timegrid, prices, 0.5, None)
        np.testing.assert_almost_equal(
            out["dispatch"]["c1"].values[0:5],
            np.array([0.625, 0.50, 0.375, 0.250, 0.125]),
            4,
        )
        np.testing.assert_almost_equal(
            out["dispatch"]["c2"].values[0:5],
            np.array([-0.625, -0.50, -0.375, -0.250, -0.125]),
            4,
        )
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[5:], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[5:], 0, 4)

    def test_simple_case_no_time_trailing(self):
        timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq="15min")
        T = timegrid.T
        prices = {
            "price_1": np.ones(T) * 2,
            "price_2": -100 * np.ones(T),
        }
        prices["price_2"][T - 5 : T] = 5  ## here, we will see dispatch c1 ---> c2
        out = self.simple_case_with_ramp(timegrid, prices, 0.5, None)
        np.testing.assert_almost_equal(
            out["dispatch"]["c1"].values[T - 5 : T],
            np.array([0.125, 0.250, 0.375, 0.500, 0.625]),
            4,
        )
        np.testing.assert_almost_equal(
            out["dispatch"]["c2"].values[T - 5 : T],
            np.array([-0.125, -0.250, -0.375, -0.5, -0.625]),
            4,
        )
        np.testing.assert_almost_equal(out["dispatch"]["c1"].values[0 : T - 5], 0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["c2"].values[0 : T - 5], 0, 4)

    def test_ramp_with_CHP_asset_or_multi(self):
        """CHP Asset does ramps differently, but in simple case should be identical"""
        timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq="2h")
        T = timegrid.T
        prices = {
            "price_1": np.ones(T) * 2,
            "price_2": -100 * np.ones(T),
        }
        prices["price_2"][5:10] = 5  ## here, we will see dispatch c1 ---> c2
        node = eao.Node("power_node")

        ## case only contracts
        c1 = eao.assets.Contract(
            name="c1", price="price_1", nodes=node, min_cap=0, max_cap=10.0, ramp=2
        )
        c2 = eao.assets.Contract(
            name="c2", price="price_2", nodes=node, min_cap=-10.0, max_cap=0
        )
        portf = eao.portfolio.Portfolio([c1, c2])
        out_c = eao.optimize(portf, timegrid, prices)
        a = out_c["dispatch"]["c1"][5:10]
        b = np.array([4, 8, 12, 8, 4])
        np.testing.assert_almost_equal(a, b, 4)
        ### case with Plant instead of c1
        p1 = eao.assets.Plant(
            name="c1", price="price_1", nodes=node, min_cap=0, max_cap=10.0, ramp=2
        )
        portf = eao.portfolio.Portfolio([p1, c2])
        out_p = eao.optimize(portf, timegrid, prices)
        a = out_p["dispatch"]["c1"][5:10]
        np.testing.assert_almost_equal(a, b, 4)
        ### case with multi
        node2 = eao.Node("extra")
        m1 = eao.assets.MultiCommodityContract(
            name="c1",
            price="price_1",
            nodes=[node, node2],
            min_cap=0,
            max_cap=10.0,
            factors_commodities=[1, 0],
            ramp=2,
        )
        portf = eao.portfolio.Portfolio([m1, c2])
        out_m = eao.optimize(portf, timegrid, prices)
        a = out_m["dispatch"]["c1 (power_node)"][5:10]
        np.testing.assert_almost_equal(a, b, 4)
        self.assertAlmostEqual(
            (out_m["dispatch"]["c1 (extra)"][5:10]).abs().sum(), 0, 3
        )

    def test_batterie_mit_rampen(self):
        """Test and illustration: Create a ramp for a battery via a contract"""
        timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq="h")
        T = timegrid.T
        prices = {"price": np.ones(T)}
        prices["price"][2::2] = 0
        ## portf
        ramp = 0.5
        node = eao.Node("power_node")
        c = eao.assets.Contract(
            name="markt",
            price="price",
            nodes=node,
            min_cap=-10,
            max_cap=10.0,
            ramp=ramp,
        )
        b = eao.assets.Storage(
            name="bat",
            nodes=node,
            cap_in=1,
            cap_out=1,
            size=4,
            start_level=2,
            end_level=2,
        )
        portf = eao.portfolio.Portfolio([c, b])
        out = eao.optimize(portf, timegrid, prices)
        a = out["dispatch"]["markt"].values
        diffs = abs(a[:-1] - a[1:])
        np.testing.assert_array_less(diffs, 0.5, ramp + 1e-4)

    def test_ramp_with_extra_cost(self):
        """extra_cost will lead to 2x the disp variables"""
        timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq="2h")
        T = timegrid.T
        prices = {
            "price_1": np.ones(T) * 2,
            "price_2": -100 * np.ones(T),
        }
        prices["price_2"][10:15] = 5  ## here, we will see dispatch c1 ---> c2
        node = eao.Node("power_node")

        ## case only contracts
        c1 = eao.assets.Contract(
            name="c1",
            price="price_1",
            nodes=node,
            min_cap=-10,
            max_cap=10.0,
            ramp=3,
            extra_costs=0.1,  ### !!!
        )
        c2 = eao.assets.Contract(
            name="c2", price="price_2", nodes=node, min_cap=-10.0, max_cap=10
        )
        portf = eao.portfolio.Portfolio([c1, c2])
        out_c = eao.optimize(portf, timegrid, prices)
        a = out_c["dispatch"]["c1"]
        b = -20 * np.ones(12)
        b[-2] = -14
        b[-1] = -8
        ### Achtung: Am Anfang darf es keine Rampe geben. Im Code mit Matrizen ist die erste Zeile
        # 1, 0, .... ---> darf nicht sein
        # Shape ist T x T ---> darf nicht sein
        # .....> erste Zeile löschen (implementiert)
        np.testing.assert_almost_equal(a, b, 3)

    def test_ramp_with_extra_cost_2(self):
        """extra_cost will lead to 2x the disp variables"""
        timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq="h")
        T = timegrid.T
        prices = {
            "price_1": np.ones(T) * 2,
            "price_2": -100 * np.ones(T),
        }
        prices["price_2"][10:15] = 5  ## here, we will see dispatch c1 ---> c2
        node = eao.Node("power_node")

        ## case only contracts
        c1 = eao.assets.Contract(
            name="c1",
            price="price_1",
            nodes=node,
            min_cap=-10,
            max_cap=10.0,
            ramp=5,
            extra_costs=0.1,  ### !!!
        )
        c2 = eao.assets.Contract(
            name="c2", price="price_2", nodes=node, min_cap=-10.0, max_cap=10
        )
        #### with extra_costs --> double the number of vars
        for ec in [0, 1]:
            c1.extra_costs = ec
            prices = {"price_1": np.ones(T) * 100, "price_2": 0 * np.ones(T)}
            prices["price_1"][5:] = 0
            prices["price_2"][5:] = 2
            portf = eao.portfolio.Portfolio([c1, c2])
            out_c = eao.optimize(portf, timegrid, prices)
            a = out_c["dispatch"]["c1"]
            b = np.array(
                [
                    -10.0,
                    -10.0,
                    -10.0,
                    -10.0,
                    -10.0,
                    -5.0,
                    0.0,
                    5.0,
                    10.0,
                    10.0,
                ]
            )
            np.testing.assert_almost_equal(a[0:10], b, 3)
            pass

    def test_batterie_mit_rampen_edge_cases(self):
        """Test and illustration: Create a ramp for a battery via a contract"""
        timegrid = eao.Timegrid(dt.date(2021, 1, 1), dt.date(2021, 1, 4), freq="15min")
        T = timegrid.T
        prices = {"price": np.linspace(1, 10, T)}
        prices["price"][0] = -1000
        ## portf
        ramp = 0.5
        node = eao.Node("power_node")
        c = eao.assets.Contract(
            name="markt",
            price="price",
            nodes=node,
            min_cap=-100,
            max_cap=100.0,
            ramp=None,
        )
        b = eao.assets.Storage(
            name="bat",
            nodes=node,
            cap_in=100,
            cap_out=100,
            size=200,
            start_level=2,
            end_level=10,
            ramp=ramp,
            cost_in=1.1,
            block_size="d",
        )
        portfA = eao.portfolio.Portfolio([c, b])
        bb = b.copy
        cc = c.copy
        bb.ramp = None
        cc.ramp = ramp
        portfB = eao.portfolio.Portfolio([cc, bb])
        outA = eao.optimize(portfA, timegrid, prices)
        outB = eao.optimize(portfB, timegrid, prices)
        flA = outA["internal_variables"]["bat_fill_level"].values
        flB = outB["internal_variables"]["bat_fill_level"].values
        dA = outA["dispatch"]["bat"].values
        dB = outB["dispatch"]["bat"].values
        diffA = abs(dA[1:] - dA[:-1])
        diffB = abs(dB[1:] - dB[:-1])
        np.testing.assert_almost_equal(dA, dB, 3)
        self.assertAlmostEqual(max(diffA), ramp / 4, 4)  # end level is min
        pass
