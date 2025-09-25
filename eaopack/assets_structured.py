### This file contains structured assets that rely on portfolio classes

import numpy as np
import abc
import scipy.sparse as sp

from typing import Union, List, Dict, Sequence, Tuple
from eaopack.assets import *
from eaopack.portfolio import Portfolio
from eaopack.optimization import OptimProblem
from eaopack.basic_classes import Timegrid, Unit, Node, StartEndValueDict, convert_time_unit


class StructuredAsset(Asset):
    """ Structured asset that wraps a portfolio in one asset
        Example: hydro storage with inflow consisting of several linked storage levels """
    def __init__(self,
                 portfolio: Portfolio,
                *args,
                **kwargs): 
        """ Structured asset that wraps a portfolio

        Args:
            portf (Portfolio): Portfolio to be wrapped
            nodes (nodes as in std. asset): where to connect the asset to the outside. 
                                            Must correspond to (a) node(s) of the internal structure
        """
        super().__init__(*args, **kwargs)
        self.portfolio = portfolio

    @abc.abstractmethod
    def setup_optim_problem(self, prices: dict, timegrid:Timegrid = None, costs_only:bool = False) -> OptimProblem:
        """ Set up optimization problem for asset

        Args:
            prices (dict): Dictionary of price arrays needed by assets in portfolio
            timegrid (Timegrid, optional): Discretization grid for asset. Defaults to None, 
                                           in which case it must have been set previously
            costs_only (bool): Only create costs vector (speed up e.g. for sampling prices). Defaults to False                                            
        Returns:
            OptimProblem: Optimization problem to be used by optimizer
        """
        # loop through assets,  set start/end and set timegrid
        if timegrid is None:
            timegrid = self.timegrid
        else: 
            self.set_timegrid(timegrid)
        for a in self.portfolio.assets:
            if not ((self.start is None) or (a.start is None)):
                a.start = max(a.start, self.start)
            if not ((self.end is None) or (a.end is None)):
                a.end   = min(a.end, self.end)
            a.set_timegrid(timegrid)
        # create optim problem, skipping creation of nodal restrictions
        #   those will be added by the overall portfolio
        #   the structured asset basically only hides away internal nodes
        #   also skip nodal restrictions for external nodes in the contract portfolio
        op = self.portfolio.setup_optim_problem(prices, timegrid, skip_nodes = self.node_names)
        if costs_only:
            return op.c
        op.mapping.rename(columns={'index_assets':'index_internal_assets_'+self.name}, inplace = True)
        # store original asset name
        op.mapping['internal_asset'] = op.mapping['asset']
        # record asset in variable name
        if 'var_name' in op.mapping.columns:
            op.mapping['var_name'] = op.mapping['var_name']+'__'+op.mapping['asset']
        # assign all variables to the struct asset
        op.mapping['asset'] = self.name
        # connect asset nodes to the outside and mark internal variables
        internal_nodes = op.mapping['node'].unique() # incl those to external
        for n in internal_nodes:
            if n not in self.node_names:
                In = op.mapping['node'] == n
                ## store information for later extraction
                op.mapping.loc[In, 'node'] = self.name+'_internal_'+str(n)
            # define variables as internal
                op.mapping.loc[In, 'type'] = 'i'
        return op

class LinkedAsset(StructuredAsset):
    """
    Linked asset that wraps a portfolio in one asset and poses additional constraints on variables.
    This can be used to ensure that one asset turns on only after another asset has been running for at
    least a set amount of time.
    """
    def __init__(self, portfolio:Portfolio,
                 asset1_variable: Tuple[Union[Asset, str], str, Union[Node, str]],
                 asset2_variable: Tuple[Union[Asset, str], str, Union[Node, str]],
                 asset2_time_already_running: Union[str, float] = "time_already_running",
                 time_back: float = 1,
                 time_forward: float = 0,
                 name: str = 'default_name_linkedAsset',
                 *args, **kwargs):
        """ Linked asset that wraps a portfolio in one asset and poses the following additional constraints on variable
        v1 of asset1 and (bool) variable v2 of asset2:

        v1_t <= u1_t * v2_{t+i}, for all i = -time_back,...,time_forward   and   timesteps t = 0,...,timegrid.T

        Here, v1_t and v2_t stand for variable v1 of asset1 at timestep t and variable v2 of asset2 at timestep t, respectively.
        u1_t stands for the upper bound for variable v1_t as specified in asset1.

        This can be used to ensure that a dispatch or "on" variable v1 is 0 (or "off") depending on the value of an "on" variable v2.
        For example, it can be ensured that asset1 only turns "on" or has positive dispatch once asset2 has
        been running for a minimum amount of time.

        Args:
            portf (Portfolio): Portfolio to be wrapped
            nodes (nodes as in std. asset): where to connect the asset to the outside.
                                            Must correspond to (a) node(s) of the internal structure
            name (str): name of the linked asset
            asset1_variable (Tuple[Union[Asset, str], str, Union[Node, str]]): Tuple specifying the variable v1 consisting of
                - asset1 (Asset, str): asset or asset_name of asset in portfolio
                - v1 (str): name of a variable in asset1
                - node1 (Node, str): node or node_name of node in portfolio
            asset2_variable (Tuple[Union[Asset, str], str, Union[Node, str]]): Tuple specifying the variable v2 consisting of
                - asset2 (Asset, str): asset or asset_name of asset in portfolio
                - v2 (str): name of a bool variable in asset2
                - node2 (Node, str): node or node_name of node in portfolio
            asset2_time_already_running (Union[str, float]): Indicating the runtime asset2 has already been running for
                float: the time in the timegrids main_time_unit that asset2 has been 'on' for
                str: the name of an attribute of asset2 that indicates the time asset2 has been running
                This defaults to "time_already_running"
            time_back(float): The minimum amount of time asset2 has to be running before v1 of asset1 can be > 0
            time_forward(float): The minimum amount of time v1 of asset1 has to be 0 before asset2  is turned off
        """

        super().__init__(portfolio=portfolio, name=name, *args, **kwargs)

        a1, v1, node1 = asset1_variable
        a2, v2, node2 = asset2_variable

        if isinstance(a1, Asset):
            self.asset1 = a1
        else:
            self.asset1 = self.portfolio.get_asset(a1)

        if isinstance(a2, Asset):
            self.asset2 = a2
        else:
            self.asset2 = self.portfolio.get_asset(a2)

        self.variable1_name = v1
        self.variable2_name = v2

        self.node1_name = node1
        if self.node1_name is not None:
            if isinstance(self.node1_name, Node):
                self.node1_name = self.node1_name.name
            if self.node1_name not in self.node_names:
                self.node1_name = self.name + '_internal_' + self.node1_name

        self.node2 = node2
        if self.node2 is not None:
            if isinstance(self.node2, Node):
                self.node2 = self.node2.name
            if self.node2 not in self.node_names:
                self.node2 = self.name + '_internal_' + self.node2

        if isinstance(asset2_time_already_running, str):
            self.asset2_time_already_running = getattr(self.asset2, asset2_time_already_running, None)
            if self.asset2_time_already_running is None:
                print("Warning: Asset", self.asset2.name, "has no attribute", asset2_time_already_running + ". "
                      "Therefore, 0 is used per default.")
                self.asset2_time_already_running = 0
        else:
            self.asset2_time_already_running = asset2_time_already_running
        self.time_back = time_back
        self.time_forward = time_forward

    def setup_optim_problem(self, prices: dict, timegrid: Timegrid = None, costs_only: bool = False) -> OptimProblem:
        """ set up optimization problem for the asset

        Args:
            prices (dict): dictionary of price np.arrays. dict must contain a key that corresponds
                            to str "price" in asset (if prices are required by the asset)
            timegrid (Timegrid): Grid to be used for optim problem. Defaults to none
            costs_only (bool): Only create costs vector (speed up e.g. for sampling prices). Defaults to False

        Returns:
            OptimProblem: Optimization problem that may be used by optimizer
        """
        op = super().setup_optim_problem(prices, timegrid, costs_only)

        # convert time_back and time_forward from timegrids main_time_unit to timegrid.freq
        time_back = self.convert_to_timegrid_freq(self.time_back, "time_back")
        time_forward = self.convert_to_timegrid_freq(self.time_forward, "time_forward")
        asset2_time_already_running = self.convert_to_timegrid_freq(self.asset2_time_already_running, "asset2_time_already_running")

        for t in range(self.timegrid.restricted.T):
            condition = (op.mapping['var_name'] == self.variable1_name + '__' + self.asset1.name) & (op.mapping["time_step"] == t)
            if self.node1_name is not None:
                condition = condition & (op.mapping["node"] == self.node1_name)
            else:
                condition = condition & (op.mapping["node"].isnull())
            I1_t = op.mapping.index[condition]
            assert I1_t[0].size == 1
            for i in np.arange(-time_back, time_forward + 1):
                if i + t < -asset2_time_already_running:
                    # asset2 has not been running long enough, so variable1 of asset1 has to be 0
                    op.u[I1_t] = 0
                    continue
                if i + t < 0 or i + t >= self.timegrid.restricted.T:
                    continue
                condition = (op.mapping['var_name'] == self.variable2_name + '__' + self.asset2.name) & (op.mapping["time_step"] == i + t)
                if self.node2 is not None:
                    condition = condition & (op.mapping["node"] == self.node2)
                else:
                    condition = condition & (op.mapping["node"].isnull())
                I2_it = op.mapping.index[condition]
                assert I2_it[0].size == 1
                a = sp.lil_matrix((1, op.A.shape[1]))
                a[0, I1_t] = 1
                a[0, I2_it] = -op.u[I1_t]
                op.A = sp.vstack((op.A, a))
                op.cType += 'U'
                op.b = np.hstack((op.b, 0))
        return op

