### This file contains renewable energy assets such as photovoltaic, wind, or batteries

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
from eaopack.assets import *


class RenewableAsset(Asset):

    def __init__(
        self,
        name: str = "default_name_renewable",
        nodes: Node = Node(name="default_node"),
        start: dt.datetime = None,
        end: dt.datetime = None,
        wacc: float = 0,
        freq: str | None = None,
        profile: float | StartEndValueDict | str = 0.0,
        controllable: bool = True,
        sell_position: bool = False,
        market_price: float | StartEndValueDict | str = 0.0,
        fixed_price: float | None = None,
        n_hour_rule_payment: int | None = None,
        n_hour_rule_delivery: int | None = None,
        cfd_type: bool = False,
    ):
        """Renewable Asset or PPA contract
            A renewable asset (PPA) produces (delivers) green power according to a given profile
            Fixed payments according to subsidy schemes or PPA terms are added

        General asset parameters:
            name (str): Unique name of the asset                                              (asset parameter)
            node (Node): Node, the contract is located in                                    (asset parameter)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used

        Overall characteristics:
            profile (float, StartEndValueDict, str, optional):     Production profile given e.g. by wind or PV availability. Defaults to 0.
            controllable (bool, optional):  Production can be regulated down to zero. Defaults to True
            sell_position (bool, optional): For PPAs - if True capacity is negative (i.e. to be delivered). Defaults to False (capacity positive)

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
        assert profile is not None, "RenewableAsset argument profile cannot be None."
        assert market_price is not None, "RenewableAsset argument market_price cannot be None."
        super().__init__(name=name, nodes=nodes, start=start, end=end, wacc=wacc, freq=freq)
        self.profile = profile
        self.fixed_price = fixed_price
        self.market_price = market_price
        self.cfd_type = cfd_type
        self.controllable = controllable
        self.sell_position = sell_position
        self.n_hour_rule_payment = n_hour_rule_payment
        self.n_hour_rule_delivery = n_hour_rule_delivery

    @abc.abstractmethod
    def setup_optim_problem(
        self,
        prices: dict | None = None,
        timegrid: Timegrid | None = None,
        costs_only: bool = False,
    ) -> OptimProblem:

        # set timegrid if given as optional argument - needed for make_vector calls
        if not timegrid is None:
            self.set_timegrid(timegrid)

        # Payment logic
        fixed_price = self.make_vector(prices=prices, value=self.fixed_price, convert=False)
        market_price = self.make_vector(prices=prices, value=self.market_price, convert=False)
        effective_price_name = "effective_price"
        if self.cfd_type:
            effective_price = {effective_price_name: fixed_price - market_price}
        else:
            effective_price = {effective_price_name: fixed_price}

        # profile is effectively the max_cap; min_cap is either 0 or equal max_cap if asset is not controllable
        max_cap = self.profile
        min_cap = self.make_vector(value=0, convert=True) if self.controllable else max_cap

        # Asset in short position; min_cap and max_cap in reverse
        if self.sell_position:
            max_cap, min_cap = -min_cap, -max_cap

        # n_hour_rules for payment and delivery (different values may apply)
        if self.n_hour_rule_payment is not None:
            n_hours = self.convert_to_timegrid_freq(
                self.n_hour_rule_payment, "n_hour_rule_payment", timegrid=timegrid)
            idx = n_hour_rule_applies(market_price, n_hours)
            effective_price[idx] = 0

        if  self.n_hour_rule_delivery is not None:
            n_hours = self.convert_to_timegrid_freq(
                self.n_hour_rule_delivery, "n_hour_rule_delivery", timegrid=timegrid)
            idx = n_hour_rule_applies(market_price, n_hours)
            min_cap[idx] = 0
            max_cap[idx] = 0

        # set up SimpleContract and call setup_optim_problem:
        internal_contract = SimpleContract(
            name=self.name,
            nodes=self.nodes,
            start=self.start,
            end=self.end,
            wacc=self.wacc,
            freq=self.freq,
            min_cap=min_cap,
            max_cap=max_cap,
            price=effective_price_name
        )
        return internal_contract.setup_optim_problem(
            prices=effective_price, timegrid=timegrid, costs_only=costs_only
        )


def n_hour_rule_applies(price: np.ndarray, n_hours: int) -> np.ndarray:
    """
    Find n_hours-long consecutive interval of negative prices and return their indexes
    """
    mask = price <= 0
    padded = np.concatenate(([False], mask, [False]))  # Pad with False at both ends to catch edge runs
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    rule_applies = lengths >= n_hours
    idx = np.concatenate(
        [
            np.arange(s, e)
            for s, e in zip(
            starts[rule_applies], ends[rule_applies]
        )
        ]
    )
    return idx
