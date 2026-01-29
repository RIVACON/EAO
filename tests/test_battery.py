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


class BatteryTest(unittest.TestCase):
    def test_optimization(self):
        """trivial test with eff_out"""
        node = eao.assets.Node("testNode")
        timegrid = eao.assets.Timegrid(
            dt.date(2021, 1, 1), dt.date(2021, 1, 2), freq="h"
        )
        a = eao.assets.Storage(
            "STORAGE",
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

    def test_no_cycles(self):
        """trivial test max number cycles"""
        node = eao.assets.Node("testNode")
        timegrid = eao.assets.Timegrid(
            dt.date(2021, 1, 1), dt.date(2021, 1, 5), freq="h"
        )
        a = eao.assets.Storage(
            "STORAGE",
            node,
            size=2,
            cap_in=1,
            cap_out=1,
            start_level=0,
            end_level=0,
            price="price",
            max_cycles_no=1.1,
            max_cycles_freq="d",
        )
        price = np.sin(np.linspace(0, 200, timegrid.T)) + 3
        prices = {"price": price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        n = int(timegrid.T)
        xin = res.x[0:n]
        xout = res.x[n : 2 * n]
        fl = a.fill_level(op, res)
        myrange = pd.date_range(
            start=timegrid.start,
            end=timegrid.end + pd.Timedelta("1d"),
            freq="d",
            inclusive="both",
        )
        for i in range(0, len(myrange) - 1):
            myI = (timegrid.timepoints >= myrange[i]) & (
                timegrid.timepoints < myrange[i + 1]
            )
            if any(myI):
                print(abs(xin[myI].sum()))
                self.assertTrue(
                    abs(xin[myI].sum()) <= a.max_cycles_no * a.size / 0.9 + 0.0001
                )

    def test_two_versions(self):
        """implementing two alternative ways - new battery and via contract max_take / reformulation of roundtrip efficiency"""
        timegrid = eao.assets.Timegrid(
            dt.date(2021, 1, 1), dt.date(2021, 1, 10), freq="2h"
        )
        np.random.seed(2709)
        buy = np.random.randn(timegrid.T).cumsum()
        sell = buy - 0.2 * np.random.rand(timegrid.T)
        prices = {"buy": buy, "sell": sell}

        ######################### settings
        battery_data = {}  # capacity, size and efficiency of an on-site battery
        battery_data["cap"] = 1  # MW
        battery_data["size"] = 2 * battery_data["cap"]  # 2 hours
        battery_data["eff_in"] = 0.8
        battery_data["eff_out"] = 0.9
        battery_data["max_roundtrip"] = 2.2
        battery_data["max_roundtrip_freq"] = "d"
        battery_data["simult_in_out"] = True
        ### Structural setup, distinguishing own assets and supply from the grid
        node_power = eao.assets.Node("behind meter")

        myrange = pd.date_range(
            start=timegrid.start,
            end=timegrid.end + pd.Timedelta("10d"),
            freq=battery_data["max_roundtrip_freq"],
            inclusive="both",
        )
        max_take = eao.StartEndValueDict(
            start=myrange.values[0:-1],
            end=myrange.values[1:],
            values=np.ones(len(myrange) - 1)
            * battery_data["max_roundtrip"]
            * battery_data["size"]
            / battery_data["eff_in"],
        )

        buy = eao.assets.SimpleContract(
            name="buy", nodes=node_power, price="buy", min_cap=0, max_cap=1000
        )
        buy_max_take = eao.assets.Contract(
            name="buy",
            nodes=node_power,
            price="buy",
            min_cap=0,
            max_cap=1000,
            max_take=max_take,
        )
        sell = eao.assets.SimpleContract(
            name="sell", nodes=node_power, price="sell", min_cap=-1000, max_cap=0
        )
        ### Our battery
        battery = eao.assets.Storage(
            name="battery",
            nodes=node_power,
            cap_in=battery_data["cap"],
            cap_out=battery_data["cap"],
            eff_in=battery_data["eff_in"] * battery_data["eff_out"],
            size=battery_data["size"] * battery_data["eff_out"],
            start_level=0.5 * battery_data["size"] * battery_data["eff_out"],
            end_level=0.5 * battery_data["size"] * battery_data["eff_out"],
            no_simult_in_out=battery_data["simult_in_out"],
            block_size="2d",
        )

        battery_new = eao.assets.Storage(
            name="battery",
            nodes=node_power,
            cap_in=battery_data["cap"],
            cap_out=battery_data["cap"],
            eff_in=battery_data["eff_in"],
            eff_out=battery_data["eff_out"],
            size=battery_data["size"],
            start_level=0.5 * battery_data["size"],
            end_level=0.5 * battery_data["size"],
            max_cycles_no=battery_data["max_roundtrip"],
            max_cycles_freq=battery_data["max_roundtrip_freq"],
            no_simult_in_out=battery_data["simult_in_out"],
            block_size="2d",
        )
        portf = eao.portfolio.Portfolio([battery, buy_max_take, sell])
        portf_new = eao.portfolio.Portfolio([battery_new, buy, sell])

        out = eao.optimize(portf=portf, timegrid=timegrid, data=prices, solver="SCIP")
        new = eao.optimize(
            portf=portf_new, timegrid=timegrid, data=prices, solver="SCIP"
        )
        self.assertAlmostEqual(
            out["summary"].loc["value", "Values"],
            new["summary"].loc["value", "Values"],
            4,
        )

        myrange = pd.date_range(
            start=timegrid.start,
            end=timegrid.end + pd.Timedelta("1d"),
            freq=battery_data["max_roundtrip_freq"],
            inclusive="both",
        )
        mymax = (
            battery_data["size"]
            * battery_data["max_roundtrip"]
            / battery_data["eff_in"]
        )
        for i in range(0, len(myrange) - 1):
            myI = (timegrid.timepoints >= myrange[i]) & (
                timegrid.timepoints < myrange[i + 1]
            )
            if any(myI):
                mysum = out["internal_variables"].loc[myI, "battery_charge"].sum()
                self.assertGreater(mymax + 1e-3, mysum)
                mysum = new["internal_variables"].loc[myI, "battery_charge"].sum()
                self.assertGreater(mymax + 1e-3, mysum)

        out["internal_variables"].loc[myI, "battery_charge"].max()
        self.assertGreater(
            battery_data["size"] + 1e-3,
            out["internal_variables"].loc[:, "battery_charge"].max(),
        )

    def test_blocks_split(self):
        """trivial test with eff_out"""
        node = eao.assets.Node("testNode")
        timegrid = eao.assets.Timegrid(
            dt.date(2021, 1, 1), dt.date(2021, 1, 5), freq="4h"
        )
        a = eao.assets.Storage(
            "STORAGE",
            node,
            size=5,
            cap_in=1,
            cap_out=1,
            start_level=0.5,
            end_level=2,
            price="price",
            eff_in=0.8,
            eff_out=0.9,
            no_simult_in_out=False,
            block_size="d",
        )
        price = np.ones([timegrid.T])
        price[0:-1:2] = 50

        prices = {"price": price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()
        xin = res.x[0 : timegrid.T]
        xout = res.x[timegrid.T :]
        fl = a.fill_level(op, res)
        # ensure start and end level
        self.assertAlmostEqual(fl[0] + xin[0] * 0.8 + xout[0] / 0.9, 0.5, 5)
        self.assertAlmostEqual(fl[-1], 2, 5)
        # ensure end level between blocks
        myi = timegrid.timepoints.hour == 20
        for i in timegrid.I[myi]:
            self.assertAlmostEqual(fl[i], 2, 5)

    def test_time_dependent_size(self):
        """test time dependency of storage size"""
        node = eao.assets.Node("testNode")
        timegrid = eao.assets.Timegrid(
            dt.date(2021, 1, 1), dt.date(2021, 1, 10), freq="h"
        )
        a = eao.assets.Storage(
            "STORAGE",
            node,
            size="size",
            cap_in=100,  # "instantaneous" charge / discharge"
            cap_out=100,
            start_level=0,
            end_level=10,
            price="price",
        )
        price = np.ones([timegrid.T])
        price[0 : timegrid.T : 2] = 0  # expect every 2nd hour empty / full
        data = {"price": price}
        data["size"] = np.linspace(1, 10, timegrid.T)
        op = a.setup_optim_problem(data, timegrid=timegrid)
        res = op.optimize()
        # every 2nd time step completely full to changing level
        np.testing.assert_almost_equal(
            -res.x.cumsum()[0 : timegrid.T : 2], data["size"][0 : timegrid.T : 2], 4
        )

    def test_time_dependent_capa_in(self):
        """test time dependency of storage capacity in"""
        node = eao.assets.Node("testNode")
        timegrid = eao.assets.Timegrid(
            dt.date(2021, 1, 1), dt.date(2021, 1, 10), freq="h"
        )
        a = eao.assets.Storage(
            "STORAGE",
            node,
            size=50,
            cap_in="capa_in",
            cap_out=100,
            start_level=0,
            end_level=10,
            cost_out=1e-3,
            price="price",
        )
        price = np.ones([timegrid.T])
        price[0 : timegrid.T : 2] = 0  # expect every 2nd hour empty / full
        data = {"price": price}
        data["capa_in"] = np.linspace(1, 10, timegrid.T)
        op = a.setup_optim_problem(data, timegrid=timegrid)
        res = op.optimize()
        # every 2nd should be full power (changing capa)
        np.testing.assert_almost_equal(
            -res.x[0 : timegrid.T : 2], data["capa_in"][0 : timegrid.T : 2], 4
        )

    def test_time_dependent_capa_out(self):
        """test time dependency of storage capacity out"""
        node = eao.assets.Node("testNode")
        timegrid = eao.assets.Timegrid(
            dt.date(2021, 1, 1), dt.date(2021, 1, 10), freq="h"
        )
        a = eao.assets.Storage(
            "STORAGE",
            node,
            size=50,
            cap_in=100,
            cap_out="capa_out",
            start_level=0,
            end_level=10,
            cost_out=1e-3,
            price="price",
        )
        price = np.ones([timegrid.T])
        price[0 : timegrid.T : 2] = 0  # expect every 2nd hour empty / full
        data = {"price": price}
        data["capa_out"] = np.linspace(2, 8, timegrid.T)
        op = a.setup_optim_problem(data, timegrid=timegrid)
        res = op.optimize()
        # every 2nd should be full power (changing capa)
        np.testing.assert_almost_equal(
            res.x[1 + timegrid.T : 2 * timegrid.T : 2],
            data["capa_out"][1 : timegrid.T : 2],
            4,
        )

    def test_specific_battery(self):
        """given portfolio. test behaviour"""

        s = """{"__class__": "Timegrid",
            "end": {
                "__class__": "datetime",
                "__tz__": "CET",
                "__value__": "2026-02-27 23:00:00"
            },
            "freq": "h",
            "main_time_unit": "h",
            "start": {
                "__class__": "datetime",
                "__tz__": "CET",
                "__value__": "2026-01-27 23:00:00"
            }
        }"""
        tg = eao.serialization.load_from_json(s)
        s = """
            {
                "__class__": "Portfolio",
                "assets": [
                    {
                        "__class__": "Asset",
                        "asset_type": "SimpleContract",
                        "end": null,
                        "extra_costs": 0.0,
                        "freq": null,
                        "max_cap": 1000,
                        "min_cap": -1000,
                        "name": "dah",
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
                        "price": "dah",
                        "start": null,
                        "wacc": 0
                    },
                    {
                        "__class__": "Asset",
                        "asset_type": "Storage",
                        "block_size": "d",
                        "cap_in": 200.0,
                        "cap_out": 200.0,
                        "cost_in": 0.0,
                        "cost_out": 0.0,
                        "cost_store": 0.0,
                        "eff_in": 1,
                        "eff_out": 1,
                        "end": null,
                        "end_level": 200.0,
                        "freq": null,
                        "inflow": 0.0,
                        "max_cycles_freq": "d",
                        "max_cycles_no": null,
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
                        "size": 400.0,
                        "start": null,
                        "start_level": 200.0,
                        "wacc": 0.0
                    }
                ]
            }        """
        ### shorten
        start = pd.Timestamp(2026, 1, 28, tz="CET")
        end = pd.Timestamp(2026, 1, 31, tz="CET")
        tg = eao.Timegrid(start, end, freq="h")
        #### basic test
        portf = eao.serialization.load_from_json(s)
        data = pd.read_pickle("tests/battery_test_data.pkl")
        # check index is correct
        out = eao.optimize(portf=portf, timegrid=tg, data=data, split_interval_size="d")
        self.assertEqual(tg.start, out["dispatch"].index[0])
        self.assertTrue(all(tg.timepoints == out["dispatch"].index))

        ### more tests
        start = pd.Timestamp(2026, 1, 28, tz="CET")
        end = pd.Timestamp(2026, 1, 29, tz="CET")
        endings = [
            pd.Timestamp(2026, 1, 29, tz="CET"),
            pd.Timestamp(2026, 1, 31, tz="CET"),
        ]
        for end in endings:
            ### situation: one day only -- expect same problem with/without blocks
            # ### parameter setting without blocks
            tg = eao.Timegrid(start, end, freq="h")
            portf.get_asset("battery").block_size = None
            portf.get_asset("battery").start_level = 100
            portf.get_asset("battery").end_level = 200
            bat = portf.get_asset("battery").copy
            op_nb = bat.setup_optim_problem(timegrid=tg)
            # out = eao.optimize(portf=portf, timegrid=tg, data=data)
            # d = out["dispatch"]
            # d["price"] = out["prices"]["input data: dah"]
            # self.assertAlmostEqual(d.loc["2026-01-28 23:00:00+01:00", "battery"], -200, 3)
            ### parameter setting WITH blocks
            portf.get_asset("battery").block_size = "d"
            bat = portf.get_asset("battery").copy
            op_wb = bat.setup_optim_problem(timegrid=tg)
            out = eao.optimize(portf=portf, timegrid=tg, data=data)
            d = out["dispatch"]
            d["price"] = out["prices"]["input data: dah"]
            self.assertAlmostEqual(
                d.loc["2026-01-28 23:00:00+01:00", "battery"], -200, 3
            )
            np.testing.assert_allclose(op_wb.b, op_nb.b, 3)
            np.testing.assert_allclose(op_wb.A.todense(), op_nb.A.todense(), 3)
            ## check fill level at end of each day
            fl = out["internal_variables"]["battery_fill_level"]
            np.testing.assert_almost_equal(
                fl[fl.index.hour == 23].values, portf.get_asset("battery").end_level, 3
            )
            # compute fill level from disp
            # eao.io.output_to_file(out, "test_output.xlsx")
            d = -out["dispatch"]["battery"]
            eff_in = portf.get_asset("battery").eff_in
            eff_out = portf.get_asset("battery").eff_out
            ffl = (np.where(d > 0, d * eff_in, d / eff_out)).cumsum() + portf.get_asset(
                "battery"
            ).start_level
            np.testing.assert_almost_equal(fl, ffl, 1)
            # test copy
            p = portf.copy
            el = portf.get_asset("battery").end_level
            el_c = p.get_asset("battery").end_level
            self.assertAlmostEqual(el, el_c, 4)
            p.end_level = 111
            portf.end_level = 222
            self.assertAlmostEqual(p.end_level, 111.0, 4)
            self.assertAlmostEqual(portf.end_level, 222.0, 4)
            pass

    def test_disp_end_block(self):
        """test there may be disp at end of block"""
        node = eao.assets.Node("testNode")
        timegrid = eao.assets.Timegrid(
            dt.date(2021, 1, 1), dt.date(2021, 1, 3), freq="2h"
        )
        a = eao.assets.Storage(
            "STORAGE",
            node,
            size=5,
            cap_in=1,
            cap_out=1,
            start_level=0.1,
            price="price",
            end_level=1,
            eff_in=1,
            eff_out=1,
            block_size="d",
        )
        c = eao.assets.SimpleContract(
            nodes=node, price="price", min_cap=-100, max_cap=100
        )
        portf = eao.Portfolio([a.copy, c])
        price = np.ones([timegrid.T])
        price[0:-1:2] = -1
        price[11] = 100  # make sth happen at end of day ... 2h freq!
        price[23] = 100  # make sth happen at end of day

        prices = {"price": price}
        op = a.setup_optim_problem(prices, timegrid=timegrid)
        res = op.optimize()  # via asset
        portf.get_asset("STORAGE").price = None
        out = eao.optimize(portf=portf, timegrid=timegrid, data=prices)  # via portf

        x = res.x
        fl = a.fill_level(op, res)
        ffl = out["internal_variables"]["STORAGE_fill_level"]
        # np.testing.assert_almost_equal(fl, ffl, 3)
        np.testing.assert_almost_equal(ffl[ffl.index.hour == 22].values, 1, 3)
        d = out["dispatch"]["STORAGE"]
        # ### check with block None for one day
        ### smth happening last hours in day? Full capa!
        np.testing.assert_almost_equal(d[d.index.hour == 22].values, 2, 3)
        np.testing.assert_almost_equal(x[d.index.hour == 22], 2, 3)
        pass


class TestBatteryWithRamp(unittest.TestCase):

    def contract_and_battery(self, timegrid, ramp_battery, ramp_contract):
        node = eao.assets.Node("testNode")
        a = eao.assets.Storage(
            "STORAGE1",
            node,
            size=50,
            cap_in=100,
            cap_out=100,
            start_level=50,
            end_level=0,
            cost_out=0,
            eff_in=0.8,
            ramp=ramp_battery,
        )
        c = eao.assets.Contract(
            name="c2", price="price", nodes=node, min_cap=-100.0, max_cap=100, ramp=ramp_contract
        )
        prices = {
            "price": np.linspace(1, 100, timegrid.T)
        }
        portf = eao.portfolio.Portfolio([a, c])
        return eao.optimize(portf, timegrid, prices)

    def test_battery_with_ramp(self):
        timegrid = eao.assets.Timegrid(
            dt.date(2021, 1, 1), dt.date(2021, 1, 10), freq="h"
        )
        T = timegrid.T
        # ramp in battery: At largest t the maximal d(dispatch)/dt = 5 is observed:
        out = self.contract_and_battery(timegrid, 5, None)
        np.testing.assert_almost_equal(out["dispatch"]["STORAGE1"].values[0:T-4], 0.0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["STORAGE1"].values[T-4:T], [5, 10, 15, 20], 4)

        # ramp in contract: At largest t the same maximal d(dispatch)/dt = 5 is observed:
        out = self.contract_and_battery(timegrid, None, 5)
        np.testing.assert_almost_equal(out["dispatch"]["STORAGE1"].values[0:T-4], 0.0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["STORAGE1"].values[T-4:T], [5, 10, 15, 20], 4)

        # No ramps: At largest t the same maximal d(dispatch)/dt = capacity is observed:
        out = self.contract_and_battery(timegrid, None, None)
        np.testing.assert_almost_equal(out["dispatch"]["STORAGE1"].values[0:T-1], 0.0, 4)
        np.testing.assert_almost_equal(out["dispatch"]["STORAGE1"].values[T:], 50, 4)

###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == "__main__":
    unittest.main()
