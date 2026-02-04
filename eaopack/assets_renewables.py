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
            min_cap: Union[float, StartEndValueDict, str] = 0.0,
            max_cap: Union[float, StartEndValueDict, str] = 0.0,
            freq: str = None,
            periodicity: str = None,
            periodicity_duration: str = None,
    ):
        """Renewable Asset: given price and limited capacity in/out. No other constraints
            A renewable asset is able to produce green energy according to a given profile at given prices plus extra subsidies to given capacity limits

        Args:
            name (str): Unique name of the asset                                              (asset parameter)
            node (Node): Node, the constract is located in                                    (asset parameter)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used

            min_cap (float, dict) : Minimum flow/capacity for buying (negative)
            max_cap (float, dict) : Maximum flow/capacity for selling (positive)
                                    float: constant value
                                    dict:  dict['start'] = array
                                           dict['end']   = array
                                           dict['values'] = array
                                    str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)
            price (str): Name of price vector for buying / selling. Defaults to None
            extra_costs (float, dict, str): extra costs added to price vector (in or out). Defaults to 0.
                                            float: constant value
                                            dict:  dict['start'] = array
                                                   dict['end']   = array
                                                   dict['values'] = array
                                            str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)

            periodicity (str, pd freq style): Makes assets behave periodically with given frequency. Periods are repeated up to freq intervals (defaults to None)
            periodicity_duration (str, pd freq style): Intervals in which periods repeat (e.g. repeat days over whole weeks)  (defaults to None)
        """
        super().__init__(  # parent: SimpleContract
            name=name,
            nodes=nodes,
            start=start,
            end=end,
            wacc=wacc,
            price=price,
            extra_costs=extra_costs,
            min_cap=min_cap,
            max_cap=max_cap,
            freq=freq,
            periodicity=periodicity,
            periodicity_duration=periodicity_duration
        )


    @abc.abstractmethod
    def setup_optim_problem(
            self,
            prices: Union[dict, None] = None,
            timegrid: Timegrid = None,
            costs_only: bool = False,
    ) -> OptimProblem:
        op = super().setup_optim_problem(  # parent: SimpleContract
            prices=prices, timegrid=timegrid, costs_only=costs_only
        )



        return op
