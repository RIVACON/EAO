### This file contains renewable energy assets such as photovoltaic, wind, or batteries

from typing import Union, List, Dict, Sequence
import datetime as dt
import abc
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import scipy.sparse as sp

# from scipy.sparse.lil import lil_matrix

from eaopack.basic_classes import (
    Timegrid,
    Unit,
    Node,
    StartEndValueDict,
    convert_time_unit,
)
from eaopack.optimization import OptimProblem
from eaopack.optimization import Results

import eaopack.assets as ea  # basic asset classes (to inherit from)


class RenewableAsset(ea.SimpleContract):
    def __init__(
            self,
            name: str = "default_name_simple_contract",
            nodes: Node = Node(name="default_node"),
            start: dt.datetime = None,
            end: dt.datetime = None,
            wacc: float = 0,
            price: str = None,
            extra_costs: Union[float, StartEndValueDict, str] = 0.0,
            freq: str = None,
            profile: Union[float, StartEndValueDict, str] = 0.0,
            subsidy: float = None,
            fixed_price: float = None,
            n_hour_rule: int = None,
            controllable: bool = False,
    ):
        """Renewable Asset: given price and limited capacity in/out. No other constraints
            A renewable asset is able to produce green energy according to a given profile at given prices plus extra subsidies to given capacity limits

            @@ Assumptions - TO BE REVIEWED @@
            - If a fixed_price is defined the subsidy must be None. extra_costs = price - fixed_price
            - If a subsidy is defined the fixed_price must be None. extra_costs = -subsidy
            - If n_hour_rule is not None, there is no dispatch during periods of zero or negative prices equal or longer than n_hour_rule. This is achieved by setting the price and extra_cost vectors for these periods to zero

        Args:
            name (str): Unique name of the asset                                              (asset parameter)
            node (Node): Node, the constract is located in                                    (asset parameter)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used
            price (str): Name of price vector for buying / selling. Defaults to None
            extra_costs (float, dict, str): extra costs added to price vector (in or out). Defaults to 0.
                                            float: constant value
                                            dict:  dict['start'] = array
                                                   dict['end']   = array
                                                   dict['values'] = array
                                            str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)
            profile (float, dict, str): Energy profile that the renewable asset converts to electric power, e.g., solar or wind. Defaults to 0.
            subsidy (float, optional): Subsidy paid on top of the market price, cannot be given together with fixed_price. Defaults to None.
            fixed_price (float, optional): Fixed price paid instead of the market price, cannot be given together with subsidy. Defaults to None.
            n_hour_rule (int, optional): Number of time intervals defining a minimum period length. If for this period market prices are zero or negative no energy is dispatched. Defaults to None.
        """
        super().__init__(  # parent: SimpleContract
            name=name,
            nodes=nodes,
            start=start,
            end=end,
            wacc=wacc,
            price=price,
            extra_costs=extra_costs,
            freq=freq,
        )
        assert ((subsidy is not None) != (fixed_price is not None)), \
            "Either subsidy or fixed_price must be specified, but not both"
        self.profile = profile
        self.subsidy = subsidy
        self.fixed_price = fixed_price
        self.n_hour_rule = n_hour_rule


    @abc.abstractmethod
    def setup_optim_problem(
            self,
            prices: Union[dict, None] = None,
            timegrid: Timegrid = None,
            costs_only: bool = False,
    ) -> OptimProblem:
        # profile is effectively the max_cap
        if self.profile is not None:
            self.max_cap = self.profile
            self.min_cap = 0  # check!

        # Get prices vector
        if self.price is not None:
            assert isinstance(self.price, str), (
                    "Error in asset " + self.name + " --> price must be given as string"
            )
            assert self.price in prices, f"Price {self.price} is not available in given prices dict"
            price = prices[self.price].copy()
        else:
            price = None

        # Case constant subsidy: Subsidy is paid on-top of price -> -subsidy are extra-costs
        if self.subsidy is not None:
            self.extra_costs = -self.subsidy

        # Case fixed price: price - extra_costs = fixed_price, i.e., extra_costs = price - fixed_price
        if self.fixed_price is not None:
            if price is not None:
                if isinstance(price, list):
                    self.extra_costs = np.asarray(price) - self.fixed_price
                else:
                    self.extra_costs = np.asarray(price) - self.fixed_price
            else:
                self.extra_costs = -self.fixed_price

        # Find n_hour_rule-long consecutive interval of negative prices and set them to zero
        if (self.n_hour_rule is not None and
                price is not None and
                timegrid is not None):
            n_hour_rule = self.convert_to_timegrid_freq(self.n_hour_rule, "n_hour_rule", timegrid=timegrid)
            mask = price <= 0
            # Pad with False at both ends to catch edge runs
            padded = np.concatenate(([False], mask, [False]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            lengths = ends - starts
            n_hour_rule_applies = lengths >= n_hour_rule
            idx = np.concatenate([
                np.arange(s, e) for s, e in zip(starts[n_hour_rule_applies], ends[n_hour_rule_applies])
            ])
            price[idx] = 0.0
            # same for extra_costs -> assumption: if subsidy or fixed_price and n_hour_rule is given there is no dispatch
            # in the n_hor_rule interval and therefore no subsidy
            if isinstance(self.extra_costs, (float, int, np.ndarray)):
                self.extra_costs = self.extra_costs * np.ones(timegrid.T)
            self.extra_costs[idx] = 0.0

        renewable_prices = {self.price: price}
        op = super().setup_optim_problem(  # parent: SimpleContract
            prices=renewable_prices, timegrid=timegrid, costs_only=costs_only
        )
        return op
