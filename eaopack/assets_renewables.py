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
        name: str = "default_name_renewable",
        nodes: Node = Node(name="default_node"),
        start: dt.datetime = None,
        end: dt.datetime = None,
        wacc: float = 0,
        price: str = None,
        extra_costs: Union[float, StartEndValueDict, str] = 0.0,
        freq: Union[
            str, None
        ] = None,  ## ToDo: bitte bei None als Default als Union angeben
        profile: Union[float, StartEndValueDict, str] = 0.0,
        subsidy: float = None,  ## ToDo: bitte bei None als Default als Union angeben
        fixed_price: float = None,  ## ToDo: bitte bei None als Default als Union angeben
        n_hour_rule: int = None,  ## ToDo: bitte bei None als Default als Union angeben
        controllable: bool = False,
    ):
        """Renewable Asset or PPA contract
            A renewable asset (PPA) produces (delivers) green power according to a given profile
            Fixed payments according to subsidy schemes or PPA terms are added

        General asset parameters:
            name (str): Unique name of the asset                                              (asset parameter)
            node (Node): Node, the constract is located in                                    (asset parameter)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used

        Overall characteristics:
            profile (float, StartEndValueDict, str, optional):     Production profile given e.g. by wind or PV availability. Defaults to 0.
            controllable (bool, optional):  Production can be regulated down to zero. Defaults to True
            sell_position (bool, optional): For PPAs - if True capacity is negative (i.e. to be deliverd). Defaults to False (capacity positive)

        Payment characteristics:
            fixed_price (float, str, StartEndValueDict, optional): Subsidy or PPA fixed payment per volume, e.g. EUR/MWh. Defaults to 0.
                                                                   fixed_price may be received (positive) or paid (negative)
            n_hour_rule_payment (int, optional):  Number of hours defining a minimum period length. If for this period market prices are zero
                                                  no fixed_price is paid. Defaults to None.
            n_hour_rule_delivery (int, optional): Rule as for n_hour_rule_payment. Here: if applies no delivery (e.g. for PPAs). Defaults to None
            market_price (float, str, StartEndValueDict, optional): Underlying market price for payment terms. Defaults to 0.
            cfd_type (bool, optional):            If True, (fixed_price-market_price) is paid. If False, fixed_price. Defaults to False


            ###############################################################
            Siehe unten
                # @@ Assumptions - TO BE REVIEWED @@
                # - If a fixed_price is defined the subsidy must be None. extra_costs = price - fixed_price
                # - If a subsidy is defined the fixed_price must be None. extra_costs = -subsidy
                # - If n_hour_rule is not None, there is no dispatch during periods of zero or negative prices equal or longer than n_hour_rule. This is achieved by setting the price and extra_cost vectors for these periods to zero

            Die Logik "overall"
            - max_cap = profile
            - if controllable: min_cap = 0 else: min_cap = max_cap
            - if sell_position sind wir das Ding short --> max_cap und min_cap tauschen und negativ

            Logik Payment:
            - payment: if cfd_type: payment = fixed_price - market_price else: payment = fixed_price
            - n_hour_rule_payment: where applies payment = 0
            - n_hour_rule_delivery: where applies max_cap = min_cap = 0

            ---> Übergabe SimpleContract
            - price = payment ## ! Achtung: Nicht, wie vorher gesagt extra_cost
            - extra_cost --> nicht übergeben

            Generell bei (float, StartEndValueDict, str) self.make_vector nutzen. Achtung: bei Leistung umrechnen auf timegrid mit convert = True!

            #### Test setups:
            * Markt mit Preis; RES Asset mit Profil; Alles default values außer market_price; Profil. Test: Be neg. Preisen wird abgeregelt
            * Markt mit Preis; RES Asset mit Profil; Alles default values außer market_price; controllable = False Profil. Test: Be neg. Preisen wird NICHT abgeregelt
            * Variante mit cfd_type --> cash flow im RES Asset korrekt "minus Marktpreis?
            * n-Stunden Regel für fixed payment und/ oder delivery
            * RESAsset mit gleichen Parametern -- eins "long", eins "short". Gleicht sich beides aus? Bewertung mit Short Asset n-Stundenregel und controllable, long asset nicht
            ...

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
        assert (subsidy is not None) != (  ### keine Unterscheidung mehr, jetzt boolean
            fixed_price is not None
        ), "Either subsidy or fixed_price must be specified, but not both"
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

        # ToDo: Use "self.make_vector" to convert to suitable array for all variables (where suited)
        ### im SimpleContract gab es noch eine alte (umständliche) Variante; ist geändert
        # Get prices vector
        if self.price is not None:
            assert isinstance(self.price, str), (
                "Error in asset " + self.name + " --> price must be given as string"
            )
            assert (
                self.price in prices
            ), f"Price {self.price} is not available in given prices dict"
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
        if self.n_hour_rule is not None and price is not None and timegrid is not None:
            n_hour_rule = self.convert_to_timegrid_freq(
                self.n_hour_rule, "n_hour_rule", timegrid=timegrid
            )
            mask = price <= 0
            # Pad with False at both ends to catch edge runs
            padded = np.concatenate(([False], mask, [False]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            lengths = ends - starts
            n_hour_rule_applies = lengths >= n_hour_rule
            idx = np.concatenate(
                [
                    np.arange(s, e)
                    for s, e in zip(
                        starts[n_hour_rule_applies], ends[n_hour_rule_applies]
                    )
                ]
            )
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
