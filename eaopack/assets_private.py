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

import eaopack.assets as ea

class PlantWithMarketRisk(ea.CHPAsset):
    def __init__(self,
                 name: str = 'default_name_plant',
                 nodes: List[Node] = [Node(name = 'default_node_power'), Node(name = 'default_node_gas_optional')],
                 start: dt.datetime = None,
                 end:   dt.datetime = None,
                 wacc: float = 0,
                 price:str = None,
                 extra_costs: Union[float, StartEndValueDict, str] = 0.,
                 min_cap: Union[float, StartEndValueDict, str] = 0.,
                 max_cap: Union[float, StartEndValueDict, str] = 0.,
                 min_take:StartEndValueDict = None,
                 max_take:StartEndValueDict = None,
                 freq: str = None,
                 profile: pd.Series = None,
                 periodicity: str = None,
                 periodicity_duration: str = None,
                 ramp: float = None,
                 start_costs: Union[float, Sequence[float], StartEndValueDict] = 0.,
                 running_costs: Union[float, StartEndValueDict, str] = 0.,
                 min_runtime: float = 0,
                 time_already_running: float = 0,
                 min_downtime: float = 0,
                 time_already_off: float = 0,
                 last_dispatch: float = 0,
                 start_ramp_lower_bounds: Sequence = None,
                 start_ramp_upper_bounds: Sequence = None,
                 shutdown_ramp_lower_bounds: Sequence = None,
                 shutdown_ramp_upper_bounds: Sequence = None,
                 ramp_freq: str = None,
                 start_fuel: Union[float, StartEndValueDict, str] = 0.,
                 fuel_efficiency: Union[float, StartEndValueDict, str] = 1.,
                 consumption_if_on: Union[float, StartEndValueDict, str] = 0.,
                 market_risk_threshold: Union[float, Sequence[float], StartEndValueDict] = 0.,
                 market_risk_costs: Union[float, Sequence[float], StartEndValueDict] = None,
                **kwargs
                 ):
        """ Plant with additional cost above a defined threshold:
            adding costs per unit when running above a certain threshold
        Args:

        Plant arguments

        additional:

        market_risk_threshold (float: optional): capacity above which additional costs apply
        market_risk_costs     (float: optional): costs that apply above a threshhold

        """
        super().__init__(name=name,
                            nodes=nodes,
                            start=start,
                            end=end,
                            wacc=wacc,
                            freq=freq,
                            price=price,
                            extra_costs=extra_costs,
                            min_cap=min_cap,
                            max_cap=max_cap,
                            min_take=min_take,
                            max_take=max_take,
                            ramp = ramp,
                            start_costs = start_costs,
                            running_costs = running_costs,
                            min_runtime = min_runtime,
                            time_already_running = time_already_running,
                            min_downtime = min_downtime,
                            time_already_off = time_already_off,
                            last_dispatch = last_dispatch,
                            start_ramp_lower_bounds = start_ramp_lower_bounds,
                            start_ramp_upper_bounds = start_ramp_upper_bounds,
                            shutdown_ramp_lower_bounds = shutdown_ramp_lower_bounds,
                            shutdown_ramp_upper_bounds = shutdown_ramp_upper_bounds,
                            ramp_freq = ramp_freq,
                            start_fuel = start_fuel,
                            fuel_efficiency = fuel_efficiency,
                            consumption_if_on = consumption_if_on,
                            _no_heat = True)
        self.market_risk_threshold = market_risk_threshold
        self.market_risk_costs     = market_risk_costs


    def setup_optim_problem(self, prices: dict, timegrid: Timegrid = None,
                            costs_only: bool = False) -> OptimProblem:
        """ Set up optimization problem for asset

        Args:
            prices (dict): Dictionary of price arrays needed by assets in portfolio
            timegrid (Timegrid, optional): Discretization grid for asset. Defaults to None,
                                           in which case it must have been set previously
            costs_only (bool): Only create costs vector (speed up e.g. for sampling prices). Defaults to False

        Returns:
            OptimProblem: Optimization problem to be used by optimizer
        """
        op = super().setup_optim_problem(prices=prices, timegrid=timegrid, costs_only=costs_only)

        market_risk_threshold = self.make_vector(self.market_risk_threshold, prices, default_value=0., convert=True)
        market_risk_costs     = self.make_vector(self.market_risk_costs, prices, default_value=0., convert = True)

        ### new part: add boolean "below threshhold" and restriction
        if (market_risk_threshold is not None) and (max(market_risk_threshold) >=0.)\
              and (market_risk_costs is not None) and (max(market_risk_costs) >=0.):

            I = self.timegrid.restricted.I # indices of restricted time grid
            T = self.timegrid.restricted.T
            max_cap = self.make_vector(self.max_cap, prices, convert=True) # Make vector of single min/max capacities.

            ###  include bool z for threshold:
            map_bool = pd.DataFrame()
            map_bool['time_step'] = I
            map_bool['node'] = np.nan
            map_bool['asset'] = self.name
            map_bool['type'] = 'i'  # internal
            map_bool['bool'] = True
            map_bool['var_name'] = 'bool_threshhold'
            map_bool.index += op.mapping.index.max()+1 # those are new variables
            op.mapping = pd.concat([op.mapping, map_bool])

            # extend A for variables
            op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))

            # set lower and upper bounds, costs:
            op.l = np.hstack((op.l, np.zeros(T)))
            op.u = np.hstack((op.u, np.ones(T)))
            op.c = np.hstack((op.c, np.zeros(T)))

            ### include disp_above to keep track of the dispatch amount above the threshold
            mapping = pd.DataFrame()
            mapping['time_step'] = I
            mapping['node'] = np.nan
            mapping['asset'] = self.name
            mapping['type'] = 'i'  # internal
            mapping['bool'] = False
            mapping['var_name'] ='disp_above' # new variable
            mapping.index += op.mapping.index.max()+1 # those are new variables
            op.mapping = pd.concat([op.mapping, mapping])

            # extend A for variables
            op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(mapping)))))

            # set lower and upper bounds, costs:
            op.l = np.hstack((op.l, np.zeros(T)))
            op.u = np.hstack((op.u, max_cap))
            op.c = np.hstack((op.c, market_risk_costs))

            ### Define restrictions
            # Collect variables from mapping
            node_power = self.nodes[0].name
            map_disp = op.mapping.loc[(op.mapping['node'] == node_power) & (op.mapping['var_name'] == 'disp'),:]
            map_disp_above = op.mapping.loc[(op.mapping['var_name'] == 'disp_above'),:]
            map_bool_thres = op.mapping.loc[(op.mapping['var_name'] == 'bool_threshhold'),:]
            assert len(map_disp)==len(map_bool), 'error- lengths of disp and bools do not match'
            assert len(map_disp)==len(map_disp_above), 'error- lengths of disp and disp_above do not match'

            ### Constraint 1:
            # disp_above_t <= M * bool_threshold_t
            # 1 * disp_above_t - M * bool_threshold_t <= 0
            myA = sp.lil_matrix((len(map_disp), op.A.shape[1]))
            i_bool = 0 # counter for booleans
            myb = np.zeros(len(map_disp))
            big_M = max(max_cap) + 1 # Big M lager than maximum dispatch

            for t in map_disp['time_step'].values:
                ind_disp_above = map_disp_above.index[map_disp['time_step'] == t][0]
                ind_bool_thres = map_bool_thres.index[map_bool_thres['time_step'] == t][0]

                myA[i_bool, ind_disp_above] = 1
                myA[i_bool, ind_bool_thres] = -big_M

                i_bool += 1

            op.A = sp.vstack((op.A, myA))
            op.cType += 'U' * (len(map_disp))
            op.b = np.hstack((op.b, myb))

            ### Constraint 2:
            # disp_above_t >= disp_t - mr_threshold_t
            # 1 * disp_t - 1 * disp_above <= mr_threshold_t
            myA = sp.lil_matrix((len(map_disp), op.A.shape[1]))
            myb = market_risk_threshold

            for t in map_disp['time_step'].values:
                ind_disp    = map_disp.index[map_disp['time_step'] == t][0]
                ind_disp_above = map_disp_above.index[map_disp['time_step'] == t][0]

                myA[t, ind_disp] = 1
                myA[t, ind_disp_above] = -1

            op.A = sp.vstack((op.A, myA))
            op.cType += 'U' * (len(map_disp))
            op.b = np.hstack((op.b, myb))

            ### Constraint 3:
            # disp_above_t <= disp_t - mr_threshold + M * (1 - bool_threshold_t)
            # -1 * disp_t + 1 * disp_above_t + M * bool_threshold_t <= (M - mr_threshold)
            myA = sp.lil_matrix((len(map_disp), op.A.shape[1]))
            i_bool = 0 # counter for booleans
            myb = np.zeros(len(map_disp)) # big_M - market_risk_threshold

            for t in map_disp['time_step'].values:
                ind_disp    = map_disp.index[map_disp['time_step'] == t][0]
                ind_disp_above = map_disp_above.index[map_disp_above['time_step'] == t][0]
                ind_bool_thres = map_bool_thres.index[map_bool_thres['time_step'] == t][0]

                myA[t, ind_disp] = -1
                myA[i_bool, ind_disp_above] = 1
                myA[i_bool, ind_bool_thres] = big_M

                myb[t] = big_M - market_risk_threshold[t]

                i_bool += 1

            op.A = sp.vstack((op.A, myA))
            op.cType += 'U' * (len(map_disp))
            op.b = np.hstack((op.b, myb))

            #print(op.A.todense())
            #print(op.b)
            #print(op.cType)

        return op

class PlantWithStepStartupCost(ea.CHPAsset):
    def __init__(self,
                 name: str = 'default_name_plant',
                 nodes: List[Node] = [Node(name = 'default_node_power'), Node(name = 'default_node_gas_optional')],
                 start: dt.datetime = None,
                 end:   dt.datetime = None,
                 wacc: float = 0,
                 price:str = None,
                 extra_costs: Union[float, StartEndValueDict, str] = 0.,
                 min_cap: Union[float, StartEndValueDict, str] = 0.,
                 max_cap: Union[float, StartEndValueDict, str] = 0.,
                 min_take:StartEndValueDict = None,
                 max_take:StartEndValueDict = None,
                 freq: str = None,
                 profile: pd.Series = None,
                 periodicity: str = None,
                 periodicity_duration: str = None,
                 ramp: float = None,
                 start_costs: Union[float, Sequence[float], StartEndValueDict] = 0.,
                 running_costs: Union[float, StartEndValueDict, str] = 0.,
                 min_runtime: float = 0,
                 time_already_running: float = 0,
                 min_downtime: float = 0,
                 time_already_off: float = 0,
                 last_dispatch: float = 0,
                 start_ramp_lower_bounds: Sequence = None,
                 start_ramp_upper_bounds: Sequence = None,
                 shutdown_ramp_lower_bounds: Sequence = None,
                 shutdown_ramp_upper_bounds: Sequence = None,
                 ramp_freq: str = None,
                 start_fuel: Union[float, StartEndValueDict, str] = 0.,
                 fuel_efficiency: Union[float, StartEndValueDict, str] = 1.,
                 consumption_if_on: Union[float, StartEndValueDict, str] = 0.,
                 rampUp_Load_1: Union[float, Sequence[float], StartEndValueDict] = 0.,
                 rampUp_Load_2: Union[float, Sequence[float], StartEndValueDict] = 0.,
                 rampUp_Load_3: Union[float, Sequence[float], StartEndValueDict] = 0.,
                 rampUp_Cost_1: Union[float, Sequence[float], StartEndValueDict] = None,
                 rampUp_Cost_2: Union[float, Sequence[float], StartEndValueDict] = None,
                 rampUp_Cost_3: Union[float, Sequence[float], StartEndValueDict] = None,
                 **kwargs):
        """ Plant with multiple start costs at diffrent steps, e.g. usefull for
            a power plant with switching of coal mills.
        Args:

        Plant arguments

        additional:

        rampUp_Load_1 (float: optional): Load point where above additional start costs apply
        rampUp_Cost_1 (float: optional): Startup costs when different load_point are exeeded
        rampUp_Load_2 (float: optional): Load point where above additional start costs apply
        rampUp_Cost_2 (float: optional): Startup costs when different load_point are exeeded
        rampUp_Load_3 (float: optional): Load point where above additional start costs apply
        rampUp_Cost_3 (float: optional): Startup costs when different load_point are exeeded
        """
        super().__init__(name=name,
                            nodes=nodes,
                            start=start,
                            end=end,
                            wacc=wacc,
                            freq=freq,
                            price=price,
                            extra_costs=extra_costs,
                            min_cap=min_cap,
                            max_cap=max_cap,
                            min_take=min_take,
                            max_take=max_take,
                            ramp = ramp,
                            start_costs = start_costs,
                            running_costs = running_costs,
                            min_runtime = min_runtime,
                            time_already_running = time_already_running,
                            min_downtime = min_downtime,
                            time_already_off = time_already_off,
                            last_dispatch = last_dispatch,
                            start_ramp_lower_bounds = start_ramp_lower_bounds,
                            start_ramp_upper_bounds = start_ramp_upper_bounds,
                            shutdown_ramp_lower_bounds = shutdown_ramp_lower_bounds,
                            shutdown_ramp_upper_bounds = shutdown_ramp_upper_bounds,
                            ramp_freq = ramp_freq,
                            start_fuel = start_fuel,
                            fuel_efficiency = fuel_efficiency,
                            consumption_if_on = consumption_if_on,
                            _no_heat = True)
        self.rampUp_Load_1 = rampUp_Load_1
        self.rampUp_Load_2 = rampUp_Load_2
        self.rampUp_Load_3 = rampUp_Load_3
        self.rampUp_Cost_1 = rampUp_Cost_1
        self.rampUp_Cost_2 = rampUp_Cost_2
        self.rampUp_Cost_3 = rampUp_Cost_3

    def setup_optim_problem(self, prices: dict, timegrid: Timegrid = None,
                            costs_only: bool = False) -> OptimProblem:
        """ Set up optimization problem for asset

        Args:
            prices (dict): Dictionary of price arrays needed by assets in portfolio
            timegrid (Timegrid, optional): Discretization grid for asset. Defaults to None,
                                           in which case it must have been set previously
            costs_only (bool): Only create costs vector (speed up e.g. for sampling prices). Defaults to False

        Returns:
            OptimProblem: Optimization problem to be used by optimizer
        """
        op = super().setup_optim_problem(prices=prices, timegrid=timegrid, costs_only=costs_only)

        # Create Vectors in time grid
        rampUp_Load_1   = self.make_vector(self.rampUp_Load_1, prices, default_value=0., convert=True)
        rampUp_Load_2   = self.make_vector(self.rampUp_Load_2, prices, default_value=0., convert=True)
        rampUp_Load_3   = self.make_vector(self.rampUp_Load_3, prices, default_value=0., convert=True)
        rampUp_Cost_1   = self.make_vector(self.rampUp_Cost_1, prices, default_value=0., convert=False)
        rampUp_Cost_2   = self.make_vector(self.rampUp_Cost_2, prices, default_value=0., convert=False)
        rampUp_Cost_3   = self.make_vector(self.rampUp_Cost_3, prices, default_value=0., convert=False)

        max_cap = self.make_vector(self.max_cap, prices, convert=True) # Make vector of single min/max capacities.
        self.big_M = (np.max(max_cap) * 4) + 1 # Big M, larger than maximum dispatch

        ## new part: add boolean variables and restrictions
        if ((rampUp_Load_1 is not None or rampUp_Load_2 is not None or rampUp_Load_3 is not None)\
            and (np.max([rampUp_Load_1, rampUp_Load_2, rampUp_Load_3]) >= 0).all()\
            and (rampUp_Cost_1 is not None or rampUp_Cost_2 is not None or rampUp_Cost_3 is not None)\
            and (np.max([rampUp_Cost_1, rampUp_Cost_2, rampUp_Cost_3]) >= 0).all()):

            ### Add binary variables for online and start/shutdown states
            op = self._add_bool_on_start_variables(op, rampUp_Cost_1, rampUp_Cost_2, rampUp_Cost_3)

            ### Define restrictions
            # Collect variables from mapping
            node_power = self.nodes[0].name
            map_disp = op.mapping.loc[(op.mapping['node'] == node_power) & (op.mapping['var_name'] == 'disp'),:]

            map_bool_on_1       = op.mapping.loc[(op.mapping['var_name'] == 'bool_on_1'),:]
            map_bool_start_1    = op.mapping.loc[(op.mapping['var_name'] == 'bool_start_1'),:]
            map_bool_on_2       = op.mapping.loc[(op.mapping['var_name'] == 'bool_on_2'),:]
            map_bool_start_2    = op.mapping.loc[(op.mapping['var_name'] == 'bool_start_2'),:]
            map_bool_on_3       = op.mapping.loc[(op.mapping['var_name'] == 'bool_on_3'),:]
            map_bool_start_3    = op.mapping.loc[(op.mapping['var_name'] == 'bool_start_3'),:]

            for i in range(1,4):
                assert len(map_disp) == len(eval(f"map_bool_on_{i}")), f'error- lengths of disp and bools do not match for bool_on_{i}'
                assert len(map_disp) == len(eval(f"map_bool_start_{i}")), f'error- lengths of disp and bools do not match for bool_start_{i}'

            # Constraints: Power comparison constraint (for all ramp up loads)
            op = self._add_constraints_for_power_comparison(op, map_disp, map_bool_on_1, map_bool_on_2, map_bool_on_3,
                                                            rampUp_Load_1, rampUp_Load_2, rampUp_Load_3)

            # Constraints: Binary step start-up constraints (for all ramp up loads)
            op = self._add_constraints_for_step_startup(op, map_bool_on_1, map_bool_on_2, map_bool_on_3,
                                                        map_bool_start_1, map_bool_start_2, map_bool_start_3)

        else:
            raise ValueError('Some arguments of PlantWithStepStartupCost are not passed right.')

        return op

    def _add_bool_on_start_variables(self, op, rampUp_Cost_1, rampUp_Cost_2, rampUp_Cost_3):
        """Add boolean variables for special step startup costs"""

        # Can probably written much shorter

        I = self.timegrid.restricted.I # indices of restricted time grid
        T = self.timegrid.restricted.T

        # Boolean online variable 1
        map_bool = pd.DataFrame()
        map_bool['time_step'] = I
        map_bool['node'] = np.nan
        map_bool['asset'] = self.name
        map_bool['type'] = 'i'  # internal
        map_bool['bool'] = True
        map_bool['var_name'] = 'bool_on_1'
        map_bool.index += op.mapping.index.max() + 1 # those are new variables
        op.mapping = pd.concat([op.mapping, map_bool])
        # extend A for variables
        op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))
        # set lower and upper bounds, costs:
        op.l = np.hstack((op.l, np.zeros(T)))
        op.u = np.hstack((op.u, np.ones(T)))
        op.c = np.hstack((op.c, np.zeros(T)))

        # Boolean online variable 2
        map_bool['var_name'] = 'bool_on_2'
        map_bool.index += len(map_bool) # those are new variables
        op.mapping = pd.concat([op.mapping, map_bool])
        # extend A for variables
        op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))
        # set lower and upper bounds, costs:
        op.l = np.hstack((op.l, np.zeros(T)))
        op.u = np.hstack((op.u, np.ones(T)))
        op.c = np.hstack((op.c, np.zeros(T)))

        # Boolean online variable 3
        map_bool['var_name'] = 'bool_on_3'
        map_bool.index += len(map_bool) # those are new variables
        op.mapping = pd.concat([op.mapping, map_bool])
        # extend A for variables
        op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))
        # set lower and upper bounds, costs:
        op.l = np.hstack((op.l, np.zeros(T)))
        op.u = np.hstack((op.u, np.ones(T)))
        op.c = np.hstack((op.c, np.zeros(T)))

        # Boolean start variable 1
        map_bool['var_name'] = 'bool_start_1'
        map_bool.index += len(map_bool) # those are new variables
        op.mapping = pd.concat([op.mapping, map_bool])
        # extend A for variables
        op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))
        # set lower and upper bounds, costs:
        op.l = np.hstack((op.l, np.zeros(T)))
        op.u = np.hstack((op.u, np.ones(T)))
        op.c = np.hstack((op.c, rampUp_Cost_1))

        # Boolean start variable 2
        map_bool['var_name'] = 'bool_start_2'
        map_bool.index += len(map_bool) # those are new variables
        op.mapping = pd.concat([op.mapping, map_bool])
        # extend A for variables
        op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))
        # set lower and upper bounds, costs:
        op.l = np.hstack((op.l, np.zeros(T)))
        op.u = np.hstack((op.u, np.ones(T)))
        op.c = np.hstack((op.c, rampUp_Cost_2))

        # Boolean start variable 3
        map_bool['var_name'] = 'bool_start_3'
        map_bool.index += len(map_bool) # those are new variables
        op.mapping = pd.concat([op.mapping, map_bool])
        # extend A for variables
        op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))
        # set lower and upper bounds, costs:
        op.l = np.hstack((op.l, np.zeros(T)))
        op.u = np.hstack((op.u, np.ones(T)))
        op.c = np.hstack((op.c, rampUp_Cost_3))

        return op

    def _add_constraints_for_power_comparison(self, op, map_disp,
                                              map_bool_on_1, map_bool_on_2, map_bool_on_3,
                                              rampUp_Load_1, rampUp_Load_2, rampUp_Load_3):
        """Add constraints for power comparison"""

        # M * (bool_on_1_t - 1) <= disp_t - ramp_Up_Load_1 <= M * bool_on_1_t
        # Split into two constraints
        # M * (bool_on_1_t - 1) <= disp_t - ramp_Up_Load_1
        # disp_t - ramp_Up_Load_1 <= big_M * bool_on_1_t
        # trantslates to
        # M * bool_on_1_t - 1 * disp_t = M - ramp_Up_Load_1
        # 1 * disp_t - M * bool_on_1_t <= ramp_Up_Load_1

        # Constraint 1.1
        # 1 * disp_t - M * bool_on_1 <= ramp_Up_Load_1
        myA = sp.lil_matrix((len(map_disp), op.A.shape[1]))
        i_bool = 0 # counter booleans
        myb = rampUp_Load_1

        for t in map_disp['time_step'].values:
            ind_disp = map_disp.index[map_disp['time_step'] == t][0]
            ind_bool_on_1 = map_bool_on_1.index[map_bool_on_1['time_step'] == t][0]

            myA[i_bool, ind_disp] = 1
            myA[i_bool, ind_bool_on_1] = -self.big_M

            i_bool += 1

        op.A = sp.vstack((op.A, myA))
        op.cType += 'U' * (len(map_disp))
        op.b = np.hstack((op.b, myb))

        # Constraint 1.2
        # M * bool_on_1_t - 1 * disp_t <= M - ramp_Up_Load_1
        myA = sp.lil_matrix((len(map_disp), op.A.shape[1]))
        i_bool = 0 # counter booleans
        myb = self.big_M - rampUp_Load_1

        for t in map_disp['time_step'].values:
            ind_disp = map_disp.index[map_disp['time_step'] == t][0]
            ind_bool_on_1 = map_bool_on_1.index[map_bool_on_1['time_step'] == t][0]

            myA[i_bool, ind_disp] = -1
            myA[i_bool, ind_bool_on_1] = self.big_M

            i_bool += 1

        op.A = sp.vstack((op.A, myA))
        op.cType += 'U' * (len(map_disp))
        op.b = np.hstack((op.b, myb))

        # Constraint 2.1
        # 1 * disp_t - M * bool_on_2 <= ramp_Up_Load_2
        myA = sp.lil_matrix((len(map_disp), op.A.shape[1]))
        i_bool = 0 # counter booleans
        myb = rampUp_Load_2

        for t in map_disp['time_step'].values:
            ind_disp = map_disp.index[map_disp['time_step'] == t][0]
            ind_bool_on_2 = map_bool_on_2.index[map_bool_on_2['time_step'] == t][0]

            myA[i_bool, ind_disp] = 1
            myA[i_bool, ind_bool_on_2] = -self.big_M

            i_bool += 1

        op.A = sp.vstack((op.A, myA))
        op.cType += 'U' * (len(map_disp))
        op.b = np.hstack((op.b, myb))

        # Constraint 2.2
        # M * bool_on_2_t - 1 * disp_t = M - ramp_Up_Load_2
        myA = sp.lil_matrix((len(map_disp), op.A.shape[1]))
        i_bool = 0 # counter booleans
        myb = self.big_M - rampUp_Load_2

        for t in map_disp['time_step'].values:
            ind_disp = map_disp.index[map_disp['time_step'] == t][0]
            ind_bool_on_2 = map_bool_on_2.index[map_bool_on_2['time_step'] == t][0]

            myA[i_bool, ind_disp] = -1
            myA[i_bool, ind_bool_on_2] = self.big_M

            i_bool += 1

        op.A = sp.vstack((op.A, myA))
        op.cType += 'U' * (len(map_disp))
        op.b = np.hstack((op.b, myb))

        # Constraint 3.1
        # 1 * disp_t - M * bool_on_3 <= ramp_Up_Load_3
        myA = sp.lil_matrix((len(map_disp), op.A.shape[1]))
        i_bool = 0 # counter booleans
        myb = rampUp_Load_3

        for t in map_disp['time_step'].values:
            ind_disp = map_disp.index[map_disp['time_step'] == t][0]
            ind_bool_on_3 = map_bool_on_3.index[map_bool_on_3['time_step'] == t][0]

            myA[i_bool, ind_disp] = 1
            myA[i_bool, ind_bool_on_3] = -self.big_M

            i_bool += 1

        op.A = sp.vstack((op.A, myA))
        op.cType += 'U' * (len(map_disp))
        op.b = np.hstack((op.b, myb))

        # Constraint 3.2
        # M * bool_on_3_t - 1 * disp_t = M - ramp_Up_Load_3
        myA = sp.lil_matrix((len(map_disp), op.A.shape[1]))
        i_bool = 0 # counter booleans
        myb = self.big_M - rampUp_Load_3

        for t in map_disp['time_step'].values:
            ind_disp = map_disp.index[map_disp['time_step'] == t][0]
            ind_bool_on_3 = map_bool_on_3.index[map_bool_on_3['time_step'] == t][0]

            myA[i_bool, ind_disp] = -1
            myA[i_bool, ind_bool_on_3] = self.big_M

            i_bool += 1

        op.A = sp.vstack((op.A, myA))
        op.cType += 'U' * (len(map_disp))
        op.b = np.hstack((op.b, myb))

        return op

    def _add_constraints_for_step_startup(self, op, map_bool_on_1, map_bool_on_2, map_bool_on_3,
                                          map_bool_start_1, map_bool_start_2, map_bool_start_3):
        """Add binary constraints for step startup costs"""

        I = self.timegrid.restricted.I # indices of restricted time grid
        T = self.timegrid.restricted.T

        # bool_on_1_t - bool_on_1_t-1 <= bool_start_1_t
        # translates to:
        # 1 * bool_on_1_t+1 - 1 * bool_on_1_t - 1 * bool_start_1_t+1 <= 0
        # (First timestep must be initialized seperatly)

        ### Cosntraint 1
        # 1 * bool_on_1_t+1 - 1 * bool_on_1_t - 1 * bool_start_1_t+1 <= 0 for all t={1,...,T-1}
        myA = sp.lil_matrix((T - 1, op.A.shape[1]))
        myb = np.zeros(T - 1)

        for t in range(T - 1):
            ind_bool_on_1 = map_bool_on_1.index[map_bool_on_1['time_step'] == t][0]
            ind_bool_start_1 = map_bool_start_1.index[map_bool_start_1['time_step'] == t][0]

            myA[t, ind_bool_on_1 + 1] = 1
            myA[t, ind_bool_on_1] = -1
            myA[t, ind_bool_start_1 + 1] = -1

        op.A = sp.vstack((op.A, myA))
        op.cType += 'U' * (T - 1)
        op.b = np.hstack((op.b, myb))

        # Initialize bool_start_1
        # If mill was not running, on means start for first time step
        # Essentially adding contraint bool_on_1 = bool_start_1 for t = 0
        if self.last_dispatch < self.rampUp_Load_1:
            ind_bool_on_1 = map_bool_on_1.index[map_bool_on_1['time_step'] == 0][0]
            ind_bool_start_1 = map_bool_start_1.index[map_bool_start_1['time_step'] == 0][0]

            a = sp.lil_matrix((1, op.A.shape[1]))
            a[0, ind_bool_on_1] = 1
            a[0, ind_bool_start_1] = -1
            op.A = sp.vstack((op.A, a))
            op.cType += 'S'
            op.b = np.hstack((op.b, 0))

        ### Constraint 2
        # 1 * bool_on_2_t+1 - 1 * bool_on_2_t - 1 * bool_start_2_t+1 <= 0 for all t={1,...,T-1}
        myA = sp.lil_matrix((T - 1, op.A.shape[1]))
        myb = np.zeros(T - 1)

        for t in range(T - 1):
            ind_bool_on_2 = map_bool_on_2.index[map_bool_on_2['time_step'] == t][0]
            ind_bool_start_2 = map_bool_start_2.index[map_bool_start_2['time_step'] == t][0]

            myA[t, ind_bool_on_2 + 1] = 1
            myA[t, ind_bool_on_2] = -1
            myA[t, ind_bool_start_2 + 1] = -1

        op.A = sp.vstack((op.A, myA))
        op.cType += 'U' * (T - 1)
        op.b = np.hstack((op.b, myb))

        # Initialize bool_start_2
        # If mill was not running, on means start for first time step
        # Essentially adding contraint bool_on_2 = bool_start_2 for t = 0
        # If last_dispatch was below mill rampUp_Load
        if self.last_dispatch < self.rampUp_Load_2:
            ind_bool_on_2 = map_bool_on_2.index[map_bool_on_2['time_step'] == 0][0]
            ind_bool_start_2 = map_bool_start_2.index[map_bool_start_2['time_step'] == 0][0]

            a = sp.lil_matrix((1, op.A.shape[1]))
            a[0, ind_bool_on_2] = 1
            a[0, ind_bool_start_2] = -1
            op.A = sp.vstack((op.A, a))
            op.cType += 'S'
            op.b = np.hstack((op.b, 0))

        ### Constraint 3
        # 1 * bool_on_3_t+1 - 1 * bool_on_3_t - 1 * bool_start_3_t+1 <= 0 for all t={1,...,T-1}
        myA = sp.lil_matrix((T - 1, op.A.shape[1]))
        myb = np.zeros(T - 1)

        for t in range(T - 1):
            ind_bool_on_3 = map_bool_on_3.index[map_bool_on_3['time_step'] == t][0]
            ind_bool_start_3 = map_bool_start_3.index[map_bool_start_3['time_step'] == t][0]

            myA[t, ind_bool_on_3 + 1] = 1
            myA[t, ind_bool_on_3] = -1
            myA[t, ind_bool_start_3 + 1] = -1

        op.A = sp.vstack((op.A, myA))
        op.cType += 'U' * (T - 1)
        op.b = np.hstack((op.b, myb))

        # Initialize bool_start_3
        # If mill was not running, on means start for first time step
        # Essentially adding contraint bool_on_3 = bool_start_3 for t = 0
        # If last_dispatch was below mill rampUp_Load
        if self.last_dispatch < self.rampUp_Load_3:
            ind_bool_on_3 = map_bool_on_3.index[map_bool_on_3['time_step'] == 0][0]
            ind_bool_start_3 = map_bool_start_3.index[map_bool_start_3['time_step'] == 0][0]

            a = sp.lil_matrix((1, op.A.shape[1]))
            a[0, ind_bool_on_3] = 1
            a[0, ind_bool_start_3] = -1
            op.A = sp.vstack((op.A, a))
            op.cType += 'S'
            op.b = np.hstack((op.b, 0))

        return op
    

class StorageClient(ea.Asset):
    """ Storage Class in Python"""
    def __init__(self,
                name    : str,
                nodes   : Node  = Node(name = 'default_node'),
                start   : dt.datetime = None,
                end     : dt.datetime = None,
                wacc    : float = 0.,
                size    : float = None,
                cap_in  : float = None,
                cap_out : float = None,
                soc_min : Union[float, str] = None,
                soc_max : Union[float, str] = None,
                start_level: float = 0.,
                end_level  : float = 0.,
                cost_out: float = 0.,
                cost_in : float = 0.,
                cost_store : float = 0.,
                block_size : str = None,
                eff_in  : float = 1.,
                eff_out  : float = 1.,
                inflow  : float = 0.,
                no_simult_in_out: bool = False,
                max_store_duration : float = None,
                price: str=None,
                freq: str = None,
                max_cycles_no: float = None,
                max_cycles_freq: str = 'd' ,
                periodicity: str = None,
                periodicity_duration: str = None                 ):
        """ Specific storage asset. A storage has the basic capability to
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

            size (float): maximum volume of commodity in storage.
            cap_in (float): Maximum flow rate for taking in a commodity
            cap_out (float): Maximum flow rate for taking in a commodity
            start_level (float, optional): Level of storage at start of optimization. Defaults to zero.
            end_level (float, optional):Level of storage at end of optimization. Defaults to zero.
            cost_out (float, optional): Cost for taking out volumes ($/volume). Defaults to 0.
            cost_in (float, optional): Cost for taking in volumes ($/volume). Defaults to 0.
            cost_store (float, optional): Cost for keeping in storage ($/volume/main time unit). Defaults to 0.
                                          Note: Cost for stored inflow is correctly optimized, but constant contribution not part of output NPV
            block_size (str, optional): Mainly to speed optimization, optimize the storage in time blocks. Defaults None (no blocks).
                                        Using pandas type frequency strings (e.g. 'd' to have a block each day)

            eff_in (float, optional): Efficiency taking in the commodity. Means e.g. at 90%: 1MWh in --> 0,9 MWh in storage. Defaults to 1 (=100%).
            eff_out: Efficiency taking out the commodity. Defaults to 1 (=100%)

            max_cycles_no   (float, optional): Maximum number of cycles the battery can perform. Defaults to None -- MIP!
            max_cycles_freq (str, optional): Frequency of the maximum number of cycles. Example: "d" for daily cycles. Defaults to 'd'

            inflow (float, optional): Constant rate of inflow volumes (flow in each time step. E.g. water inflow in hydro storage). Defaults to 0.
            no_simult_in_out (boolean, optional): Enforce no simultaneous dispatch in/out in case of costs or efficiency!=1. Makes problem MIP. Defaults to False
            max_store_duration (float, optional): Maximal duration in main time units that charged commodity can be held. Makes problem a MIP. Defaults to none

            periodicity (str, pd freq style): Makes assets behave periodicly with given frequency. Periods are repeated up to freq intervals (defaults to None)
            periodicity_duration (str, pd freq style): Intervals in which periods repeat (e.g. repeat days ofer whole weeks)  (defaults to None)
        """
        super(StorageClient, self).__init__(name=name, nodes=nodes, start=start, end=end, wacc=wacc, freq = freq)
        assert size is not None, 'Storage --'+self.name+'--: size must be given'
        self.size = size
        self.start_level = start_level
        self.end_level= end_level
        assert start_level <= size, 'Storage --'+self.name+'--: start level must be <=  storage size'
        self.cap_in = cap_in
        self.cap_out = cap_out
        assert cap_in  >=0, 'Storage --'+self.name+'--: cap_in must not be negative'
        assert cap_out >=0, 'Storage --'+self.name+'--: cap_out must not be negative'
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.eff_in = eff_in
        self.eff_out = eff_out
        self.inflow = inflow
        self.cost_out = cost_out
        self.cost_in = cost_in
        self.cost_store = cost_store
        self.price = price
        self.block_size = None
        if block_size is not None:
            self.block_size = block_size # defines the block size (as pandas frequency)
        assert len(self.nodes)<=2, 'for storage only one or two nodes valid'
        self.no_simult_in_out   = no_simult_in_out
        self.max_store_duration = max_store_duration
        self.max_cycles_no      = max_cycles_no
        self.max_cycles_freq    = max_cycles_freq
        #### periodicity
        assert not ((periodicity_duration is not None) and (periodicity is None)), 'Cannot have periodicity duration not none and periodicity none'
        self.periodicity          = periodicity
        self.periodicity_duration = periodicity_duration

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
        # set timegrid if given as optional argument
        if not timegrid is None:
            self.set_timegrid(timegrid)
        # check: timegrid set?
        assert hasattr(self, 'timegrid'), 'Set timegrid of asset before creating optim problem. Asset: '+ self.name

        dt =  self.timegrid.restricted.dt

        if len(dt) == 0: # no overlap between timegrids, asset not active
            return OptimProblem(c=np.array([]),l=np.array([]), u=np.array([]), cType='', mapping =  pd.DataFrame(),
                                timegrid = self.timegrid)
        n = self.timegrid.restricted.T # moved to Timegrid

        ct = self.cap_out * dt #  Adjust capacity (unit is in vol/h)
        cp = self.cap_in * dt  #  Adjust capacity (unit is in vol/h)
        inflow  = np.cumsum(self.inflow*dt)
        discount = self.timegrid.restricted.discount_factors

        if self.price is not None:
            assert (self.price in prices)
            price = prices[self.price].copy()
            if not (len(price) == self.timegrid.T): # price vector must have right size for discretization
                raise ValueError('Length of price array must be equal to length of time grid. Asset: '+ self.name)
            # check: if the restricted timegrid has minor and major grids, need
            # to do average over prices across minor grids
            if hasattr(self.timegrid.restricted, 'I_minor_in_major'):
                myprice = []
                #if self.profile is not None: raise NotImplementedError('Need to extend to non flat profiles')
                for myI in self.timegrid.restricted.I_minor_in_major:
                    myprice.append(price[myI].mean())
                price = np.asarray(myprice)
            else: # simply restrict prices to  asset time window
                price           = price[self.timegrid.restricted.I]
            # warning if there are neg. prices and no_simult_in_out is not True
            if (not self.no_simult_in_out) and ((self.eff_in < 1) or (self.eff_out < 1)) and (not (price >= 0).all()):
                print('Storage --'+self.name+'--: no_simult_in_out is set to False, but there are neg. prices. Likely storage will load & unload simultaneously. You want that?')
        # separation into in/out needed?  Only one or two dispatch variables per time step
        # new separation reason: separate nodes in and out
        sep_needed =  (self.eff_in != 1) or (self.eff_out != 1) or (self.cost_in !=0) or (self.cost_out !=0) or (len(self.nodes)==2) or (self.max_cycles_no is not None)
        # cost_store -- costs for keeping quantity in storage
        # effectively, for each time step t we have:   cost_store * sum_{i<t}(disp_i)
        # and after summing ofer time steps t we get   cost_store * sum_t(disp_t * N_t)
        # (discount needs to be accounted for as well)
        #       where N_t is the number of time steps after (t)
        # convert to costs per main time unit

        # Make vector of soc_min and soc_max:
        if self.soc_min is not None:
            soc_min = self.make_vector(self.soc_min, prices)
        else:
            soc_min = self.make_vector(0.0, prices)
        assert (soc_min >= 0.0).any(), 'Storage --'+self.name+'--: soc_min must not be negative'
        assert (soc_min <= self.size).any(), 'Storage --'+self.name+'--: soc_min must not be greater than size'

        if self.soc_max is not None:
            soc_max = self.make_vector(self.soc_max, prices)
        else:
            soc_max = self.make_vector(self.size, prices)
        assert (soc_max >= 0.0).any(), 'Storage --'+self.name+'--: soc_max must not be negative'
        assert (soc_max <= self.size).any(), 'Storage --'+self.name+'--: soc_max must not be greater than size'

        if self.cost_store != 0:
            cost_store = self.cost_store * dt * discount
            cost_store = np.asarray([cost_store[ii:].sum() for ii in range(0,len(cost_store))] )
        # costs in and out
        if sep_needed:
            u = np.hstack(( np.zeros(n,float), ct))
            l = np.hstack((-cp, np.zeros(n,float)))
            c = np.ones((2,n), float)
            c[0,:] = -c[0,:]*self.cost_in
            c[1,:] =  c[1,:]*self.cost_out
            if self.price is not None:
                c -= np.asarray(price)
            c = c * (np.tile(discount, (2,1)))
            if self.cost_store != 0:
                c -= (np.vstack((cost_store*self.eff_in, cost_store/self.eff_out)))
        else:
            u = ct
            l = -cp
            c = np.zeros(n)
            if self.price is not None:
                c -= np.asarray(price)*discount
            if self.cost_store != 0:
                c -= cost_store
        c  = c.flatten('C') # make all one columns
        # switch to return costs only
        if costs_only:
            return c
        # Storage restriction --  cumulative sums must fit into reservoir
        ## treatment eff_in: only eff_in reaches storage
        ## treatment of eff_out: discharge disaptch up to cap_out -- but storage is drained cap_out/eff_out
        if self.block_size is None:
            A = -sp.tril(np.ones((n,n),float))
            # Maximum: max volume not exceeded
            b = soc_max-self.start_level - inflow
            b[-1] = self.end_level - self.start_level   - inflow[-1]
            # Minimum: empty
            b_min     =  soc_min-self.start_level - inflow
            b_min[-1] =   self.end_level - self.start_level - inflow[-1]
        else:
            A = sp.lil_matrix((n,n))
            b = np.empty(n)
            b.fill(np.nan)
            b_min = np.empty(n)
            b_min.fill(np.nan)
            ### identify blocks in time grid
            try:
                buffer = pd.Timedelta(self.block_size)
            except:
                buffer = pd.Timedelta(1, self.block_size)
            indBlocks = pd.date_range(start = self.timegrid.restricted.start - buffer,
                                      end   = self.timegrid.restricted.end, freq=self.block_size)
            aa = [] # collects last element of each block
            for myd in indBlocks:
                my_bool = self.timegrid.restricted.timepoints <= myd
                if any(my_bool):
                    aa.append(np.argwhere(my_bool)[-1,-1]) # last element of block
                else:
                    aa.append(0)
                if all(my_bool): break # stop early
            aa = np.unique(np.asarray(aa))
            if aa[-1]!=n: # last block not full - last element defined to be last time step
                aa = np.append(aa,n)
            for i,a in enumerate(aa[0:-1]): # go through the blocks
                diff = aa[i+1]-a # number of elements to next block
                A[a:a+diff, a:a+diff] = - sp.tril(np.ones((diff,diff),float)) # triangle matrix for time block

                ### for first block start value is start_level
                # for subsequent blocks, start value is end value (of the last block)
                not_last = diff+a < n # the last timepoint is not included -- > start of next block is end of last
                not_first = a > 0 # first timepoint is not included --> end of last block is start of next
                if not_first:
                    my_start = self.end_level
                else:
                    my_start = self.start_level
                # Maximum: max volume not exceeded
                parts_b = (soc_max[a:a+diff]-my_start)- inflow[a:a+diff]
                parts_b[-1] = self.end_level - my_start      - inflow[a+diff-1]
                b[a:a+diff] = parts_b
                # Minimum: empty
                parts_b_min     =  soc_min[a:a+diff]-my_start  - inflow[a:a+diff]
                parts_b_min[-1] =   self.end_level - my_start - inflow[a+diff-1]
                b_min[a:a+diff] = parts_b_min
        if sep_needed:
            A = sp.hstack((A*self.eff_in, A/self.eff_out )) # for in and out
        # join restrictions for in, out, full, empty
        b = np.hstack((b, b_min))
        A = sp.vstack((A, A))
        cType = 'U'*n + 'L'*n

        ## add restrictions for max_cycles
        # quantity behind no of cycles
        if self.max_cycles_no is not None:
            cycle_quant = self.max_cycles_no * self.size
            ## create daterange for start / end of cycles
            try:   extra_time = pd.Timedelta(self.max_cycles_freq)
            except:extra_time = pd.Timedelta(1,self.max_cycles_freq)

            myrange = pd.date_range(start = self.timegrid.restricted.start,
                                    end = self.timegrid.restricted.end + extra_time,
                                    freq = self.max_cycles_freq,
                                    inclusive = 'both')
            ## loop through intervals
            for i in range(0,len(myrange)-1):
                myI = (self.timegrid.restricted.timepoints >= myrange[i]) & (self.timegrid.restricted.timepoints < myrange[i+1])
                if any(myI):
                    myA = sp.lil_matrix((1,2*n))
                    myA[0,self.timegrid.restricted.I[myI]] = -self.eff_in # restriction only for disp_in - referring to effective storage size (disp_in is negative!)
                    A = sp.vstack((A, myA))
                    b = np.hstack((b, cycle_quant))
                    cType += 'U'

        mapping = pd.DataFrame()
        if sep_needed:
            mapping['time_step'] = np.hstack((self.timegrid.restricted.I, self.timegrid.restricted.I))
            mapping['var_name']  = np.nan # name variables for use e.g. in RI
            mapping['var_name'] = mapping['var_name'].astype(str)
            ind_var_name = mapping.columns.get_indexer(['var_name'])[0]
            mapping.iloc[0:n, ind_var_name] = 'disp_in'
            mapping.iloc[n:, ind_var_name] = 'disp_out'
            if len(self.nodes)==1:
                mapping['node']      = self.nodes[0].name
            else: # separate nodes in / out.
                mapping['node']  = np.nan
                mapping['node'] = mapping['node'].astype(str)
                my_ind = mapping.columns.get_indexer(['node'])[0]
                mapping.iloc[0:n, my_ind]      = self.nodes[0].name
                mapping.iloc[n:2*n, my_ind]    = self.nodes[1].name
        else:
            mapping['time_step'] = self.timegrid.restricted.I
            mapping['node']      = self.nodes[0].name
            mapping['var_name']  = 'disp'
        mapping['asset']     = self.name
        mapping['type']      = 'd'

        ### in case of forcing no_simult_in_out - add binary variables and restrictions
        if (self.no_simult_in_out) and (sep_needed): # without sep_needed no need for forcing
            mapping['bool']      = False
            # n new binary variables
            map_bool = pd.DataFrame()
            map_bool['time_step'] = self.timegrid.restricted.I
            map_bool['node']      = np.nan
            map_bool['asset']     = self.name
            map_bool['type']      = 'i' # internal
            map_bool['bool']      = True
            map_bool['var_name']  = 'bool_1'
            mapping = pd.concat([mapping, map_bool])
            mapping.reset_index(inplace=True, drop = True) # need to reset index (which enumerates variables)
            # extend costs
            c = np.hstack((c, np.zeros(n)))
            l = np.hstack((l, np.zeros(n)))
            u = np.hstack((u, np.ones(n)))
            # extend A for binary variables (not relevant in exist. restrictions)
            # in:  (1-b)*min <= in  <= 0
            # out:        0  <= out <= (b) * max
            myn = A.shape[0] # current number of rows
            A = sp.hstack((A, sp.lil_matrix((myn,n)) ))
            # create extra restrictions
            myA = sp.lil_matrix((n,3*n))
            # "0" means mode "in"
            myA[0:n, 0:n]     = sp.eye(n)
            myA[0:n, 2*n:3*n] = sp.diags(-cp, 0)
            A                 = sp.vstack((A, myA))
            b                 = np.hstack((b, -cp))
            cType += 'L'*n
            # "1" means mode "out"
            myA = sp.lil_matrix((n,3*n))
            myA[0:n, n:2*n]     = sp.eye(n)
            myA[0:n, 2*n:3*n]   = sp.diags(-ct, 0)
            A   = sp.vstack((A, myA))
            b = np.hstack((b, np.zeros(n)))
            cType += 'U'*n


        ### in case of max_store_duration - add binary variables and restrictions
        if not self.max_store_duration is None: # without sep_needed no need for forcing
            if 'bool' not in mapping:
                mapping['bool']      = False
            # n new binary variables ... indicating that fill level is not equal to zero
            map_bool = pd.DataFrame()
            map_bool['time_step'] = self.timegrid.restricted.I
            map_bool['node']      = np.nan
            map_bool['asset']     = self.name
            map_bool['type']      = 'i' # internal
            map_bool['bool']      = True
            map_bool['var_name']  = 'bool_2'
            mapping = pd.concat([mapping, map_bool])
            mapping.reset_index(inplace=True, drop = True) # need to reset index (which enumerates variables)
            # extend costs
            c = np.hstack((c, np.zeros(n)))
            l = np.hstack((l, np.zeros(n)))
            u = np.hstack((u, np.ones(n)))
            # extend A for binary variables (not relevant in exist. restrictions)
            (n_exist,m) = A.shape
            # (1) reformulate fill level restrictions and extend A with bool ("is filled") variables
            #      replace   (Ax <= b)  by  (Ax)i - bool_i*b  <=  0
            #      n restrictions for max fill level
            A = sp.hstack((A, sp.vstack((sp.diags(-b[0:n],0),sp.lil_matrix((n_exist-n,n)) )) ))
            b[0:n] = 0
            # (2) create extra restrictions for booleans ("1 --> fill level non zero") - one for each time step
            # -->  all windows of size max_duration (md) plus one, sum of vars is <= md
            for myi in range(0,n):
                myI = (dt[myi:].cumsum()<=self.max_store_duration) # those fall into time window
                if len(np.where(~myI)[0])!=0: # full interval left
                    myI[np.where(~myI)[0][0]] = True
                    myA = sp.lil_matrix((1,m + n))
                    myA[0,np.where(myI)[0]+m+myi] = 1
                    A   = sp.vstack((A, myA))
                    b = np.hstack((b,myA.sum()-1))
                    cType += 'U'   # at most md elements may be one == fill level not md+1 times non-zero)
        # if we're using a less granular asset timegrid, add dispatch for every minor grid point
        # Effectively we concat the mapping for each minor point (one row each)
        if hasattr(self.timegrid.restricted, 'I_minor_in_major'):
            mapping = self.__extend_mapping_to_minor_grid__(mapping)

        return OptimProblem(c=c,l=l, u=u, A=A, b=b, cType=cType, mapping = mapping,
                                periodic_period_length = self.periodicity,
                                periodic_duration      = self.periodicity_duration,
                                timegrid               = self.timegrid)

    def fill_level(self, optim_problem:OptimProblem, results:Results) -> np.array:
        """ Calculate fill level of the storage incl. efficiencies etc

        Args:
            optim_problem (OptimProblem): optimization problem created by this asset
            results (Results): Results given by optimizer

        Returns:
            np.array: array with fill level per time step as per timegrid of asset
        """

        ######### missing: mapping in optim problem
        fill_level = np.zeros(self.timegrid.T)
        # filter for right asset in case larger problem is given
        my_mapping =  optim_problem.mapping.loc[(optim_problem.mapping['asset']==self.name) & (optim_problem.mapping['type']=='d')].copy()
        # drop duplicate index - since mapping may contain several rows per variable (indexes enumerate variables)
        my_mapping = pd.DataFrame(my_mapping[~my_mapping.index.duplicated(keep = 'first')])

        fill_level = np.zeros(self.timegrid.T)
        for i, r in my_mapping.iterrows():
            fill_level[r['time_step']] +=  max(0,-results.x[i])*self.eff_in \
                                         + min(0,-results.x[i])/self.eff_out
        fill_level = fill_level.cumsum() + self.start_level
        return fill_level

