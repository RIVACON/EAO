### This file contains renewable energy assets such as photovoltaic, wind, or batteries

import datetime as dt
import abc
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import scipy.sparse as sp

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
from typing import Union


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
        short_position: bool = False,
        market_price: float | StartEndValueDict | str = 0.0,
        fixed_price: float | None = None,
        n_hour_rule_payment: int | None = None,
        n_hour_rule_delivery: int | None = None,
        cfd_type: bool = False,
    ):
        """Renewable Asset or PPA contract
            A renewable asset (PPA) produces (delivers) green power according to a given profile
            Fixed payments according to subsidy schemes or PPA terms are added

        Args:
            name (str): Unique name of the asset
            node (Node): Node, the contract is located in
            start (dt.datetime) : start of asset being active. defaults to none
            end (dt.datetime)   : end of asset being active. defaults to none
            wacc (float): Weighted average cost of capital to discount cash flows in target
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq). The more granular frequency of portf & asset is used
            profile (float, StartEndValueDict, str, optional): Production profile given e.g. by wind or PV availability. Defaults to 0.
            controllable (bool, optional): Production can be regulated down to zero. Defaults to True
            short_position (bool, optional): For PPAs - if True, capacity is negative (i.e. to be delivered). Defaults to False
            fixed_price (float, str, StartEndValueDict, optional): Subsidy or PPA fixed payment per volume, e.g. EUR/MWh. Defaults to 0. Convention: cost for production: positive payment/ subsidy: negative
            market_price (float, str, StartEndValueDict, optional): Underlying market price for payment terms. Defaults to 0.
            n_hour_rule_payment (int, optional):  Number of hours defining a minimum period length. If for this period market prices are below zero. No fixed_price is paid. Defaults to None.
            n_hour_rule_delivery (int, optional): Rule as for n_hour_rule_payment. Here: if applies no delivery (e.g. for PPAs). Defaults to None
            cfd_type (bool, optional):If True, (fixed_price + market_price) is paid. If False, fixed_price. Defaults to False. Attention: convention is subsidy payment is negative (thus effectively difference to marpet price)
        """

        assert profile is not None, "RenewableAsset argument profile cannot be None."
        assert (
            market_price is not None
        ), "RenewableAsset argument market_price cannot be None."
        super().__init__(
            name=name, nodes=nodes, start=start, end=end, wacc=wacc, freq=freq
        )
        self.profile = profile
        self.fixed_price = fixed_price
        self.market_price = market_price
        self.cfd_type = cfd_type
        self.controllable = controllable
        self.short_position = short_position
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
        fixed_price = self.make_vector(
            prices=prices, value=self.fixed_price, convert=False
        )
        market_price = self.make_vector(
            prices=prices, value=self.market_price, convert=False
        )
        if self.cfd_type:
            effective_price = fixed_price + market_price
        else:
            effective_price = fixed_price

        # profile is effectively the max_cap; min_cap is either 0 or equal max_cap if asset is not controllable
        max_cap = self.profile
        min_cap = (
            self.make_vector(value=0, convert=True) if self.controllable else max_cap
        )

        # Asset in short position; min_cap and max_cap in reverse
        if self.short_position:
            max_cap, min_cap = -min_cap, -max_cap

        # n_hour_rules for payment and delivery (different values may apply)
        if self.n_hour_rule_payment is not None:
            n_hours = self.convert_to_timegrid_freq(
                self.n_hour_rule_payment, "n_hour_rule_payment", timegrid=timegrid
            )
            idx = n_hour_rule_applies(market_price, n_hours)
            effective_price[idx] = 0

        if self.n_hour_rule_delivery is not None:
            n_hours = self.convert_to_timegrid_freq(
                self.n_hour_rule_delivery, "n_hour_rule_delivery", timegrid=timegrid
            )
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
            price=effective_price,  # attention: in our logic, positive: we PAY for production, neg: subsidy
        )
        return internal_contract.setup_optim_problem(
            prices=None, timegrid=timegrid, costs_only=costs_only
        )


def n_hour_rule_applies(price: np.ndarray, n_hours: int) -> np.ndarray:
    """
    Find n_hours-long consecutive interval of negative prices and return their indexes
    """
    mask = price < 0
    padded = np.concatenate(
        ([False], mask, [False])
    )  # Pad with False at both ends to catch edge runs
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    lengths = ends - starts
    rule_applies = lengths >= n_hours
    idx = np.concatenate(
        [np.arange(s, e) for s, e in zip(starts[rule_applies], ends[rule_applies])]
    )
    return idx


class Battery(Storage):
    """Asset class for a battery (BESS), based on the Storage class.
    No additional functionality, but a simplified version omitting atypical parameters
    """

    def __init__(
        self,
        name: str,
        nodes: Node = Node(name="default_node"),
        start: Union[dt.datetime, pd.Timestamp, None] = None,
        end: Union[dt.datetime, pd.Timestamp, None] = None,
        wacc: float = 0.0,
        size: Union[float, StartEndValueDict, str] = 0.0,
        max_level: Union[float, str, StartEndValueDict, None] = None,
        min_level: Union[float, str, StartEndValueDict, None] = None,
        cap_in: Union[float, StartEndValueDict, str, None] = None,
        cap_out: Union[float, StartEndValueDict, str, None] = None,
        start_level: Union[float, StartEndValueDict, str] = 0.0,
        end_level: Union[float, StartEndValueDict, str, None] = 0.0,
        cost_out: float = 0.0,
        cost_in: float = 0.0,
        block_size: Union[None, str] = None,
        eff_in: float = 1.0,
        eff_out: float = 1.0,
        ramp_up: Union[float, StartEndValueDict, str, None] = None,
        ramp_down: Union[float, StartEndValueDict, str, None] = None,
        no_zero_transition_within_ramp=False,
        inflow: float = 0.0,
        no_simult_in_out: bool = False,
        price: Union[None, str] = None,
        freq: Union[None, str] = None,
        max_cycles_no: Union[None, float] = None,
        max_cycles_freq: str = "D",
    ):
        """Specific storage asset. A storage has the basic capability to
            (1) take in a commodity within a limited flow rate (capacity)
            (2) store a maximum volume of a commodity (size)
            (3) give out the commodity within a limited flow rate

        Args:
            name (str): Unique name of the asset (asset parameter)
            node (Node): Node, the storage is located in (asset parameter)
                         Two nodes may be defined in case input and output are located in different nodes [node_input, node_output]
            timegrid (Timegrid): Timegrid for discretization (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used

            size (float, str, StartEndValueDict): Maximum volume of commodity in storage. Use str or StartEndValueDict for time dependency
                                                  In case max_level is used, it overrides the size.
            max_level (float, str, StartEndValueDict, None): Maximum fill level. Use str or StartEndValueDict for time dependency Defaults to None (zero)
            min_level (float, str, StartEndValueDict, None): Minimum fill level. Use str or StartEndValueDict for time dependency Defaults to None (zero)

            cap_in (float, str, StartEndValueDict): Maximum flow rate for taking in a commodity. Use str or StartEndValueDict for time dependency
            cap_out (float, str, StartEndValueDict): Maximum flow rate for taking in a commodity. Use str or StartEndValueDict for time dependency
            start_level (float, str, StartEndValueDict): Level of storage at start of optimization. Defaults to zero.
            end_level (float, str, StartEndValueDict): Level of storage at end of optimization. Defaults to zero.
                                                       In case block-wise optimization or split_optimization is used, a start or end_level in form of a time series
                                                       may be useful: In this case, the respective start or end_level of the respective point in time is used
            cost_out (float, optional): Cost for taking out volumes ($/volume). Defaults to 0.
            cost_in (float, optional): Cost for taking in volumes ($/volume). Defaults to 0.
            block_size (str, optional): Mainly to speed optimization, optimize the storage in time blocks. Defaults None (no blocks).
                                        Using pandas type frequency strings (e.g. 'D' to have a block each day)

            eff_in (float, optional): Efficiency taking in the commodity. Means e.g. at 90%: 1MWh in --> 0,9 MWh in storage. Defaults to 1 (=100%).
            eff_out: Efficiency taking out the commodity. Defaults to 1 (=100%)

            max_cycles_no   (float, optional): Maximum number of cycles the battery can perform. Defaults to None -- MIP!
                                               In case a time-dependent size is chosen, max_cycles_no refers to the storage mean size in that cycle interval
            max_cycles_freq (str, optional): Frequency of the maximum number of cycles. Example: "D" for daily cycles. Defaults to 'D'

            inflow (float, str, StartEndValueDict, optional): Inflow volumes (flow in each time step. E.g. water inflow in hydro storage). Defaults to 0.
            no_simult_in_out (boolean, optional): Enforce no simultaneous dispatch in/out in case of costs or efficiency!=1. Makes problem MIP. Defaults to False
            max_store_duration (float, optional): Maximal duration in main time units that charged commodity can be held. Makes problem a MIP. Defaults to none

            ramp_up (float, StartEndValueDict, str):   Maximum increase of virtual dispatch in flow/main_time_unit (e.g. MW/h). May be time dependent
            ramp_down (float, StartEndValueDict, str): Maximum decrease of virtual dispatch in flow/main_time_unit (e.g. MW/h). May be time dependent
                                                       For ramps up or down: Positive value expected. Defaults to None. First point in time is unrestricted
                                                       ramp is given for change in flow (e.g. capacity). Since the dispatch is given in volume per time interval (e.g. MWh for 1h)
                                                       the change is given by the change in the quantity
                                                       Ramp fast: Missing is the trangle until the max. capacity is reached (capa*dt - capa^2/ramp/2)
                                                       Ramp slow: We do not reach max capacity within dt and obtain a dispatch of 0.5 * ramp * dt^2
            no_zero_transition_within_ramp (bool): If True, no zero transition in power is allowed within the ramp time. E.g. for ramp_up: if we have a positive infeed at time t,
                                                    we need to have a positive infeed at time t+1 until the ramp up time is over. Defaults to False.
        """
        super().__init__(
            name=name,
            nodes=nodes,
            start=start,
            end=end,
            wacc=wacc,
            size=size,
            max_level=max_level,
            min_level=min_level,
            cap_in=cap_in,
            cap_out=cap_out,
            start_level=start_level,
            end_level=end_level,
            cost_out=cost_out,
            cost_in=cost_in,
            block_size=block_size,
            eff_in=eff_in,
            eff_out=eff_out,
            ramp_up=ramp_up,
            ramp_down=ramp_down,
            no_zero_transition_within_ramp=no_zero_transition_within_ramp,
            inflow=inflow,
            no_simult_in_out=no_simult_in_out,
            price=price,
            freq=freq,
            max_cycles_no=max_cycles_no,
            max_cycles_freq=max_cycles_freq,
        )
