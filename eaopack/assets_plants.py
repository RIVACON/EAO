### This file contains complex power plant EAO assets

from typing import Union, List, Dict, Sequence
import datetime as dt
import abc
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import scipy.sparse as sp
# from scipy.sparse.lil import lil_matrix

from eaopack.basic_classes import Timegrid, Unit, Node, StartEndValueDict, convert_time_unit
from eaopack.optimization import OptimProblem
from eaopack.optimization import Results

import eaopack.assets as ea # basic asset classes (to inherit from)

class CHPAsset(ea.Contract):
    def __init__(self,
                 name: str = 'default_name_contract',
                 nodes: List[Node] = [Node(name = 'default_node_power'), Node(name = 'default_node_heat'), Node(name = 'default_node_gas_optional')],
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
                 conversion_factor_power_heat: Union[float, StartEndValueDict, str] = 1.,
                 max_share_heat: Union[float, StartEndValueDict, str] = None,
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
                 start_ramp_lower_bounds_heat: Sequence = None,
                 start_ramp_upper_bounds_heat: Sequence = None,
                 shutdown_ramp_lower_bounds_heat: Sequence = None,
                 shutdown_ramp_upper_bounds_heat: Sequence = None,
                 ramp_freq: str = None,
                 start_fuel: Union[float, StartEndValueDict, str] = 0.,
                 fuel_efficiency: Union[float, StartEndValueDict, str] = 1.,
                 consumption_if_on: Union[float, StartEndValueDict, str] = 0.,
                 _no_heat = False
                 ):
        """ CHPContract: Generate heat and power
            Restrictions
            - time dependent capacity restrictions
            - MinTake & MaxTake for a list of periods
            - start costs
            - minimum runtime
            - ramps
        Args:
            name (str): Unique name of the asset                                              (asset parameter)
            nodes (Node): One node each for generated power and heat                          (asset parameter)
                          optional: node for fuel (e.g. gas)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used
            profile (pd.Series, optional):  If freq(asset) > freq(portf) assuming this profile for granular dispatch (e.g. scaling hourly profile to week).
                                            Defaults to None, only relevant if freq is not none
            min_cap (float) : Minimum capacity for generating virtual dispatch (power + conversion_factor_power_heat * heat). Has to be greater or equal to 0. Defaults to 0.
            max_cap (float) : Maximum capacity for generating virtual dispatch (power + conversion_factor_power_heat * heat). Has to be greater or equal to 0. Defaults to 0.
            min_take (float) : Minimum volume within given period. Defaults to None
            max_take (float) : Maximum volume within given period. Defaults to None
                              float: constant value
                              dict:  dict['start'] = np.array
                                     dict['end']   = np.array
                                     dict['values"] = np.array
            price (str): Name of price vector for buying / selling
            extra_costs (float, dict, str): extra costs added to price vector (in or out). Defaults to 0.
                                            float: constant value
                                            dict:  dict['start'] = array
                                                   dict['end']   = array
                                                   dict['values"] = array
                                            str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)
            periodicity (str, pd freq style): Makes assets behave periodicly with given frequency. Periods are repeated up to freq intervals (defaults to None)
            periodicity_duration (str, pd freq style): Intervals in which periods repeat (e.g. repeat days ofer whole weeks)  (defaults to None)
            conversion_factor_power_heat (float, dict, str): Conversion efficiency from heat to power. Defaults to 1.
            max_share_heat (float, dict, str): Defines upper bound for the heat dispatch as a percentage of the power dispatch. Defaults to 1.
            ramp (float): Maximum increase/decrease of virtual dispatch (power + conversion_factor_power_heat * heat) in one main time unit). Defaults to None.
            start_costs (float): Costs for starting. Defaults to 0.
            running_costs (float): Costs when on. Defaults to 0.
            min_runtime (int): Minimum runtime in timegrids main_time_unit. (start ramp time and shutdown ramp time do not count towards the min runtime.) Defaults to 0.
            time_already_running (int): The number of timesteps the asset is already running in timegrids main_time_unit. Defaults to 0.
            min_downtime (int): Minimum downtime in timegrids main_time_unit. Defaults to 0.
            time_already_off (int): The number of timesteps the asset has already been off in timegrids main_time_unit. Defaults to 0.
            last_dispatch (float): Previous virtual dispatch (power + conversion_factor_power_heat * heat). Defaults to 0.
            start_ramp_lower_bounds (Sequence): The i-th element of this sequence specifies a lower bound of the
                                                virtual dispatch (power + conversion_factor_power_heat * heat) at i timesteps
                                                of freq ramp_freq after starting.  Defaults to None.
            start_ramp_upper_bounds (Sequence): The i-th element of this sequence specifies an upper bound of the
                                                virtual dispatch (power + conversion_factor_power_heat * heat) at i timesteps
                                                of freq ramp_freq after starting.  Defaults to None.
            shutdown_ramp_lower_bounds (Sequence): The i-th element of this sequence specifies a lower bound of the
                                                   virtual dispatch (power + conversion_factor_power_heat * heat) at i timesteps
                                                   of freq ramp_freq before turning off. Defaults to None.
            shutdown_ramp_upper_bounds (Sequence): The i-th element of this sequence specifies an upper bound of the
                                                   virtual dispatch (power + conversion_factor_power_heat * heat) at i timesteps
                                                   of freq ramp_freq before turning off. If it is None, it is set equal to shutdown_ramp_upper_bounds.
                                                   Defaults to None.
            start_ramp_lower_bounds_heat (Sequence): The i-th element of this sequence specifies a lower bound of heat dispatch at i timesteps
            start_ramp_upper_bounds_heat (Sequence): as above
            shutdown_ramp_lower_bounds_heat (Sequence): as above
            shutdown_ramp_upper_bounds_heat (Sequence): as above
            ramp_freq (str): A string specifying the frequency of the start-and shutdown ramp specification.
                             If this is None, the timegrids main_time_unit is used. Otherwise the start and shutdown ramps are
                             interpolated to get values in the timegrids freq.


            Optional: Explicit fuel consumption (e.g. gas) for multi-commodity simulation
                 start_fuel (float, dict, str): detaults to  0
                 fuel_efficiency (float, dict, str): defaults to 1
                 consumption_if_on (float, dict, str): defaults to 0

            _no_heat (optional):  No heat node given (making CHP a plain power plant). Defaults to False
        """
        super(CHPAsset, self).__init__(name=name,
                                       nodes=nodes,
                                       start=start,
                                       end=end,
                                       wacc=wacc,
                                       freq=freq,
                                       profile=profile,
                                       price=price,
                                       extra_costs=extra_costs,
                                       min_cap=min_cap,
                                       max_cap=max_cap,
                                       min_take=min_take,
                                       max_take=max_take,
                                       periodicity=periodicity,
                                       periodicity_duration=periodicity_duration)
        self._no_heat = _no_heat
        # check and record meaning of nodes
        self.idx_nodes = {}
        self.idx_nodes['power']        = 0
        if not _no_heat:
            assert (len(self.nodes) in (2,3)), 'Length of nodes has to be 2 or 3; power, heat and optionally fuel. Asset: ' + self.name
            self.idx_nodes['heat']     = 1
            if len(self.nodes)==3:
                self.idx_nodes['fuel']  = 2
            else:
                self.idx_nodes['fuel']  = None
        else:
            self.idx_nodes['heat']     = None
            if len(self.nodes)==2:
                self.idx_nodes['fuel']  = 1
            else:
                self.idx_nodes['fuel']  = None

        self.conversion_factor_power_heat                = conversion_factor_power_heat
        self.max_share_heat                 = max_share_heat
        self.ramp                 = ramp
        self.start_costs          = start_costs
        self.running_costs        = running_costs
        self.min_runtime          = min_runtime
        assert self.min_runtime >= 0, "Min_runtime cannot be < 0. Asset: " + self.name
        self.time_already_running = time_already_running
        self.min_downtime = min_downtime
        self.time_already_off = time_already_off
        self.last_dispatch = last_dispatch
        self.start_ramp_lower_bounds = start_ramp_lower_bounds
        self.start_ramp_upper_bounds = start_ramp_upper_bounds
        if self.start_ramp_upper_bounds is None:
            self.start_ramp_upper_bounds = self.start_ramp_lower_bounds
        assert self.start_ramp_lower_bounds is None or len(self.start_ramp_lower_bounds) == len(self.start_ramp_upper_bounds), "start_ramp_lower_bounds and start_ramp_upper_bounds cannot have different lengths. Asset: " + self.name
        self.start_ramp_time = len(self.start_ramp_lower_bounds) if self.start_ramp_lower_bounds is not None else 0
        assert np.all([self.start_ramp_lower_bounds[i] <= self.start_ramp_upper_bounds[i] for i in range(self.start_ramp_time)]), "shutdown_ramp_lower_bounds is higher than shutdown_ramp_upper bounds at some point. Asset: " + self.name
        self.shutdown_ramp_lower_bounds = shutdown_ramp_lower_bounds
        self.shutdown_ramp_upper_bounds = shutdown_ramp_upper_bounds
        if self.shutdown_ramp_upper_bounds is None:
            self.shutdown_ramp_upper_bounds = self.shutdown_ramp_lower_bounds
        assert self.shutdown_ramp_lower_bounds is None or len(self.shutdown_ramp_lower_bounds) == len(self.shutdown_ramp_upper_bounds), "start_ramp_lower_bounds and start_ramp_upper_bounds cannot have different lengths. Asset: " + self.name
        self.shutdown_ramp_time = len(self.shutdown_ramp_lower_bounds) if self.shutdown_ramp_lower_bounds is not None else 0
        assert np.all([self.shutdown_ramp_lower_bounds[i] <= self.shutdown_ramp_upper_bounds[i] for i in range(self.shutdown_ramp_time)]), "shutdown_ramp_lower_bounds is higher than shutdown_ramp_upper bounds at some point. Asset: " + self.name

        # heat start and shutdown ramps
        self.start_ramp_lower_bounds_heat = start_ramp_lower_bounds_heat
        self.start_ramp_upper_bounds_heat = start_ramp_upper_bounds_heat
        self.shutdown_ramp_lower_bounds_heat = shutdown_ramp_lower_bounds_heat
        self.shutdown_ramp_upper_bounds_heat = shutdown_ramp_upper_bounds_heat
        ### Asserts
        if shutdown_ramp_upper_bounds_heat is not None:
            assert len(shutdown_ramp_lower_bounds_heat) == len(shutdown_ramp_lower_bounds), 'shutdown lower ramp needs to habe same length for heat and power'
            assert len(shutdown_ramp_upper_bounds_heat) == len(shutdown_ramp_upper_bounds), 'shutdown upper ramp needs to habe same length for heat and power'
        if start_ramp_upper_bounds_heat is not None:
            assert len(start_ramp_upper_bounds_heat) == len(start_ramp_upper_bounds), 'start upper ramp needs to habe same length for heat and power'        
            assert len(start_ramp_lower_bounds_heat) == len(start_ramp_lower_bounds), 'start lower ramp needs to habe same length for heat and power'        

        self.ramp_freq = ramp_freq

        if (self.idx_nodes['fuel'] is not None):
            self.fuel_efficiency      = fuel_efficiency
            self.consumption_if_on    = consumption_if_on
            self.start_fuel           = start_fuel

        if self.min_downtime > 1:
            assert (self.time_already_off == 0) ^ (self.time_already_running == 0), "Either time_already_off or time_already_running has to be 0, but not both. Asset: " + self.name

    def setup_optim_problem(self, prices: dict, timegrid: Union[Timegrid, None] = None,
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

        if self.freq is not None and self.freq != self.timegrid.freq:
            raise ValueError('Freq of asset' + self.name + ' is ' + str(self.freq) + ' which is unequal to freq ' + self.timegrid.freq + ' of timegrid. Asset: ' + self.name)

        # convert min_runtime and time_already_running from timegrids main_time_unit to timegrid.freq
        min_runtime = self.convert_to_timegrid_freq(self.min_runtime, "min_runtime")
        time_already_running = self.convert_to_timegrid_freq(self.time_already_running, "time_already_running")
        min_downtime = self.convert_to_timegrid_freq(self.min_downtime, "min_downtime")
        time_already_off = self.convert_to_timegrid_freq(self.time_already_off, "time_already_off")

        # Convert start ramp and shutdown ramp from ramp_freq to timegrid.freq
        ramp_freq = self.ramp_freq
        if ramp_freq is None:
            ramp_freq = timegrid.main_time_unit
        start_ramp_time = self.start_ramp_time
        start_ramp_lower_bounds = self.start_ramp_lower_bounds
        start_ramp_upper_bounds = self.start_ramp_upper_bounds
        start_ramp_lower_bounds_heat = self.start_ramp_lower_bounds_heat
        start_ramp_upper_bounds_heat = self.start_ramp_upper_bounds_heat
        shutdown_ramp_time = self.shutdown_ramp_time        
        shutdown_ramp_lower_bounds = self.shutdown_ramp_lower_bounds
        shutdown_ramp_upper_bounds = self.shutdown_ramp_upper_bounds
        shutdown_ramp_lower_bounds_heat = self.shutdown_ramp_lower_bounds_heat
        shutdown_ramp_upper_bounds_heat = self.shutdown_ramp_upper_bounds_heat
        conversion_factor = convert_time_unit(value=1, old_freq=timegrid.freq, new_freq=timegrid.main_time_unit)
        if self.start_ramp_time:
            # power
            start_ramp_lower_bounds = self._convert_ramp(self.start_ramp_lower_bounds, ramp_freq)
            start_ramp_upper_bounds = self._convert_ramp(self.start_ramp_upper_bounds, ramp_freq)
            start_ramp_time = len(start_ramp_lower_bounds)
            start_ramp_lower_bounds = start_ramp_lower_bounds * conversion_factor
            start_ramp_upper_bounds = start_ramp_upper_bounds * conversion_factor
            # heat
            if start_ramp_lower_bounds_heat is not None:
                start_ramp_lower_bounds_heat = self._convert_ramp(self.start_ramp_lower_bounds_heat, ramp_freq)
                start_ramp_upper_bounds_heat = self._convert_ramp(self.start_ramp_upper_bounds_heat, ramp_freq)
                # start_ramp_time = len(start_ramp_lower_bounds)
                start_ramp_lower_bounds_heat = start_ramp_lower_bounds_heat * conversion_factor
                start_ramp_upper_bounds_heat = start_ramp_upper_bounds_heat * conversion_factor

        if self.shutdown_ramp_time:
            # power
            shutdown_ramp_lower_bounds = self._convert_ramp(self.shutdown_ramp_lower_bounds, ramp_freq)
            shutdown_ramp_upper_bounds = self._convert_ramp(self.shutdown_ramp_upper_bounds, ramp_freq)
            shutdown_ramp_time = len(shutdown_ramp_lower_bounds)
            shutdown_ramp_lower_bounds = shutdown_ramp_lower_bounds * conversion_factor
            shutdown_ramp_upper_bounds = shutdown_ramp_upper_bounds * conversion_factor
            # heat
            if shutdown_ramp_lower_bounds_heat is not None:
                shutdown_ramp_lower_bounds_heat = self._convert_ramp(self.shutdown_ramp_lower_bounds_heat, ramp_freq)
                shutdown_ramp_upper_bounds_heat = self._convert_ramp(self.shutdown_ramp_upper_bounds_heat, ramp_freq)
                # shutdown_ramp_time = len(shutdown_ramp_lower_bounds)
                shutdown_ramp_lower_bounds_heat = shutdown_ramp_lower_bounds_heat * conversion_factor
                shutdown_ramp_upper_bounds_heat = shutdown_ramp_upper_bounds_heat * conversion_factor

        min_runtime += start_ramp_time + shutdown_ramp_time

        # scale ramp and last dispatch in case timegrid.freq and timegrid.main_time_unit are not equal
        ramp = self.ramp * self.timegrid.restricted.dt[0] if self.ramp is not None else None
        last_dispatch = self.last_dispatch * self.timegrid.restricted.dt[0]

        # Make vectors of input params:
        start_costs = self.make_vector(self.start_costs, prices, default_value=0.)
        running_costs = self.make_vector(self.running_costs, prices, default_value=0., convert=True)
        max_share_heat = self.make_vector(self.max_share_heat, prices, default_value=1.)
        conversion_factor_power_heat = self.make_vector(self.conversion_factor_power_heat, prices, default_value=1.)
        if (self.idx_nodes['heat'] is not None): # heat note given
            assert np.all(conversion_factor_power_heat != 0), 'conversion_factor_power_heat must not be zero. Asset: ' + self.name
        if (self.idx_nodes['fuel'] is not None):
            start_fuel = self.make_vector(self.start_fuel, prices, default_value=0.)
            fuel_efficiency = self.make_vector(self.fuel_efficiency, prices, default_value=1.)
            consumption_if_on = self.make_vector(self.consumption_if_on, prices, default_value=0., convert=True)
            assert np.all(fuel_efficiency != 0), 'fuel efficiency must not be zero. Asset: ' + self.name

        # calculate costs:
        if costs_only:
            c = op
        else:
            c = op.c
        if (self.idx_nodes['heat'] is not None): # if heat node given
            c = np.hstack([c, conversion_factor_power_heat * c])  # costs for power and heat dispatch

        include_shutdown_variables = shutdown_ramp_time > 0 or start_ramp_time > 0
        include_start_variables = min_runtime > 1 or np.any(start_costs != 0) or start_ramp_time > 0 or shutdown_ramp_time > 0
        include_on_variables = include_start_variables or min_downtime > 1 or include_shutdown_variables or np.any(self.min_cap != 0.)
        if (self.idx_nodes['fuel'] is not None):
            include_start_variables = include_start_variables or np.any(start_fuel != 0.)
            include_on_variables = include_on_variables or include_start_variables or np.any(consumption_if_on != 0.)

        if include_on_variables:
            c = np.hstack([c, running_costs])  # add costs for on variables
        if include_start_variables:
            c = np.hstack([c, start_costs])  # add costs for start variables
        if include_shutdown_variables:
            c = np.hstack([c, np.zeros(self.timegrid.restricted.T)])  # costs for shutdown are 0
        if costs_only:
            return c
        op.c = c

        # Check that min_cap and max_cap are >= 0
        min_cap = op.l.copy()
        max_cap = op.u.copy()
        assert np.all(min_cap >= 0.), 'min_cap has to be greater or equal to 0. Asset: ' + self.name
        assert np.all(max_cap >= 0.), 'max_cap has to be greater or equal to 0. Asset: ' + self.name

        # Check that if include_on_variables is True, the minimum capacity is not 0. Otherwise the "on" variables cannot be computed correctly.
        if np.any(min_cap == 0) and include_on_variables:
            print("Warning for asset " + self.name + ": The minimum capacity is 0 at some point and 'on'-variables are included" 
                  ". This can lead to incorrect 'on' and 'start' variables. "
                  "To prevent this either set min_cap>0 or set min_runtime=0 and start_costs=0 and start_fuel=0"
                  " and consumption_if_on=0.")

        # Prepare matrix A:
        self.n = len(min_cap)
        if op.A is None:
            op.A = sp.lil_matrix((0, self.n))
            op.cType = ''
            op.b = np.zeros(0)

        # set lower bound to zero if I have on variables
        if include_on_variables:
            op.l = 0.*op.l # now I can be off
        # Define the dispatch variables:
        if (self.idx_nodes['heat'] is not None): # heat note given        
            op = self._add_dispatch_variables(op, conversion_factor_power_heat, max_cap, max_share_heat)

        # Add on-, start-, and shutdown-variables:
        op = self._add_bool_variables(op, include_on_variables, include_start_variables, include_shutdown_variables)

        # Minimum and maximum capacity:
        op = self._add_constraints_for_min_and_max_cap(op, min_cap, max_cap, time_already_running,
                                                       conversion_factor_power_heat, include_on_variables, start_ramp_time,
                                                       start_ramp_lower_bounds, start_ramp_upper_bounds, shutdown_ramp_time,
                                                       shutdown_ramp_lower_bounds, shutdown_ramp_upper_bounds,
                                                       start_ramp_lower_bounds_heat, start_ramp_upper_bounds_heat, 
                                                       shutdown_ramp_lower_bounds_heat, shutdown_ramp_upper_bounds_heat)

        # Ramp constraints:
        op = self._add_constraints_for_ramp(op, ramp, conversion_factor_power_heat, time_already_running,
                                            include_on_variables, max_cap, start_ramp_time, shutdown_ramp_time, last_dispatch)

        # Start and shutdown constraints:
        op = self._add_constrains_for_start_and_shutdown(op, time_already_running, include_start_variables, include_shutdown_variables)

        # Minimum runtime:
        op = self._add_constraints_for_min_runtime(op, min_runtime, include_start_variables, time_already_running)

        # Minimum Downtime:
        op = self._add_constraints_for_min_downtime(op, min_downtime, time_already_off)

        # Boundaries for the heat variable:
        if (self.idx_nodes['heat'] is not None): # heat note given        
            op = self._add_constraints_for_heat(op, max_share_heat)

        # Reset mapping index:
        op.mapping.reset_index(inplace=True, drop=True)  # need to reset index (which enumerates variables)

        # Model fuel consumption:
        if (self.idx_nodes['fuel'] is not None):
            op = self._add_fuel_consumption(op, fuel_efficiency, consumption_if_on, start_fuel, conversion_factor_power_heat, include_on_variables, include_start_variables)

        return op

    def _convert_ramp(self, ramp, ramp_freq, timegrid=None):
        """ Change the timepoints of the ramp from ramp_freq to the timegrids freq """
        if timegrid is None:
            timegrid = self.timegrid
        if ramp_freq == timegrid.freq:
            return np.array(ramp)
        converted_time = convert_time_unit(value=1, old_freq=timegrid.freq, new_freq=ramp_freq)
        if converted_time < 1:
            # timegrid.freq is finer than ramp_freq => interpolate
            ramp_duration = len(ramp)
            old_timepoints_in_new_freq = [self.convert_to_timegrid_freq(time_value=i+1, attribute_name="ramp", old_freq=ramp_freq, timegrid=timegrid, round=False) for i in
                                          range(ramp_duration)]
            new_timepoints = np.arange(np.ceil(self.convert_to_timegrid_freq(time_value=ramp_duration, attribute_name="ramp_duration", old_freq=ramp_freq, timegrid=timegrid, round=False))) + 1
            ramp_new_freq = np.interp(new_timepoints, old_timepoints_in_new_freq, ramp)
            return ramp_new_freq
        else:
            # ramp_freq is finer than timegrid.freq => use average
            ramp_padded = ramp + [ramp[-1]] * int(np.ceil(converted_time))
            new_ramp_duration = int(np.ceil(self.convert_to_timegrid_freq(len(ramp), "ramp_duration", ramp_freq, timegrid, round=False)))
            ramp_new_freq = np.zeros(new_ramp_duration)
            for i in range(ramp_new_freq.shape[0]):
                start_idx = i * converted_time
                start_idx_rounded = int(np.ceil(start_idx))
                stop_idx = (i + 1) * converted_time
                stop_idx_rounded = int(np.floor(stop_idx))
                ramp_value = 0
                if start_idx_rounded < stop_idx_rounded:
                    ramp_value = np.average(ramp_padded[start_idx_rounded: stop_idx_rounded]) * (stop_idx_rounded-start_idx_rounded)
                if start_idx_rounded > start_idx:
                    ramp_value += (start_idx_rounded - start_idx) * ramp_padded[start_idx_rounded-1]
                if stop_idx > stop_idx_rounded:
                    ramp_value += (stop_idx - stop_idx_rounded)*ramp_padded[stop_idx_rounded]
                ramp_new_freq[i] = ramp_value / (stop_idx - start_idx)

            return ramp_new_freq

    def _add_dispatch_variables(self, op, conversion_factor_power_heat, max_cap, max_share_heat):
        """ Divide each dispatch variable in op into a power dispatch that flows into the power node self.nodes[0]
            and a heat dispatch that flows into self.nodes[1] """
        # Make sure that op.mapping contains only dispatch variables (i.e. with type=='d')
        var_types = op.mapping['type'].unique()
        assert np.all(
            var_types == 'd'), "Only variables of type 'd' (i.e. dispatch variables) are allowed in op.mapping at this point. " \
                               "However, there are variables with types " + str(
            var_types[var_types != 'd']) + " in the mapping." \
                                           "This is likely due to a change in a superclass."

        self.heat_idx = len(op.mapping)

        # Divide each dispatch variable in power and heat:
        new_map = pd.DataFrame()
        for i, mynode in enumerate(self.nodes):
            if i >= 2: continue  # do only for power and heat
            initial_map = op.mapping[op.mapping['type'] == 'd'].copy()
            initial_map['node'] = mynode.name
            new_map = pd.concat([new_map, initial_map.copy()])
        op.mapping = new_map
        op.A = sp.hstack([op.A, sp.coo_matrix(conversion_factor_power_heat * op.A.toarray())])

        # Set lower and upper bounds
        op.l = np.zeros(op.A.shape[1])
        if max_share_heat is not None:
            u_heat = max_share_heat * max_cap
        else:
            u_heat = max_cap / conversion_factor_power_heat
        op.u = np.hstack((max_cap, u_heat))

        return op

    def _add_bool_variables(self, op, include_on_variables, include_start_variables, include_shutdown_variables):
        """ Add the bool variables for 'on', 'start' and 'shutdown' to the OptimProblem op if needed """
        # Add on variables
        if include_on_variables:
            self.on_idx = len(op.mapping)
            op.mapping['bool'] = False
            map_bool = pd.DataFrame()
            map_bool['time_step'] = self.timegrid.restricted.I
            map_bool['node'] = np.nan
            map_bool['asset'] = self.name
            map_bool['type'] = 'i'  # internal
            map_bool['bool'] = True
            map_bool['var_name'] = 'bool_on'
            op.mapping = pd.concat([op.mapping, map_bool])

            # extend A for on variables (not relevant in exist. restrictions)
            op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))

            # set lower and upper bounds:
            op.l = np.hstack((op.l, np.zeros(self.timegrid.restricted.T)))
            op.u = np.hstack((op.u, np.ones(self.timegrid.restricted.T)))

            # Add start variables
            if include_start_variables:
                self.start_idx = len(op.mapping)
                map_bool['var_name'] = 'bool_start'
                op.mapping = pd.concat([op.mapping, map_bool])

                # extend A for start variables (not relevant in exist. restrictions)
                op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))

                # set lower and upper bounds:
                op.l = np.hstack((op.l, np.zeros(self.timegrid.restricted.T)))
                op.u = np.hstack((op.u, np.ones(self.timegrid.restricted.T)))

            # Add shutdown variables
            if include_shutdown_variables:
                self.shutdown_idx = len(op.mapping)
                map_bool['var_name'] = 'bool_shutdown'
                op.mapping = pd.concat([op.mapping, map_bool])

                # extend A for shutdown variables (not relevant in exist. restrictions)
                op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))

                # set lower and upper bounds:
                op.l = np.hstack((op.l, np.zeros(self.timegrid.restricted.T)))
                op.u = np.hstack((op.u, np.ones(self.timegrid.restricted.T)))

        return op

    def _add_constraints_for_min_and_max_cap(self, op, min_cap, max_cap, time_already_running,
                                             conversion_factor_power_heat, include_on_variables, start_ramp_time,
                                             start_ramp_lower_bounds, start_ramp_upper_bounds, shutdown_ramp_time,
                                             shutdown_ramp_lower_bounds, shutdown_ramp_upper_bounds,
                                             start_ramp_lower_bounds_heat, start_ramp_upper_bounds_heat, 
                                             shutdown_ramp_lower_bounds_heat, shutdown_ramp_upper_bounds_heat):
        """ Add the constraints for the minimum and maximum capacity to op.

            These ensure that the virtual dispatch
            (power + conversion_factor_power_heat * heat) is 0 when the asset is "off",
            it is bounded by the start or shutdown specifications during the start and shutdown ramp,
            and otherwise it is between minimum and maximum capacity"""
        # Minimum and maximum capacity:
        start = max(0, start_ramp_time - time_already_running) if time_already_running > 0 else 0
        A_lower_bounds = sp.lil_matrix((self.n, op.A.shape[1]))
        A_upper_bounds = sp.lil_matrix((self.n, op.A.shape[1]))
        starting_timestep = self.timegrid.restricted.I[0]
        for i in range(start, self.n):
            var = op.mapping.iloc[i]

            A_lower_bounds[i, i] = 1
            if (self.idx_nodes['heat'] is not None): # heat note given        
                A_lower_bounds[i, self.heat_idx + i] = conversion_factor_power_heat[i] 
            if include_on_variables:
                A_lower_bounds[i, self.on_idx + var["time_step"] - starting_timestep] = - min_cap[i]

            A_upper_bounds[i, i] = 1
            if (self.idx_nodes['heat'] is not None): # heat note given        
                A_upper_bounds[i, self.heat_idx + i] = conversion_factor_power_heat[i] 
            if include_on_variables:
                A_upper_bounds[i, self.on_idx + var["time_step"] - starting_timestep] = - max_cap[i]

            for j in range(start_ramp_time):
                if i - j < 0:
                    continue
                A_lower_bounds[i, self.start_idx + i - j] = min_cap[i] - start_ramp_lower_bounds[j]
                A_upper_bounds[i, self.start_idx + i - j] = max_cap[i] - start_ramp_upper_bounds[j]

            for j in range(shutdown_ramp_time):
                if i + j + 1 >= self.timegrid.restricted.T:
                    break
                A_lower_bounds[i, self.shutdown_idx + i + j + 1] = min_cap[i] - shutdown_ramp_lower_bounds[j]
                A_upper_bounds[i, self.shutdown_idx + i + j + 1] = max_cap[i] - shutdown_ramp_upper_bounds[j]

        op.A = sp.vstack((op.A, A_lower_bounds[start:]))
        op.cType += 'L' * (self.n - start)
        op.b = np.hstack((op.b, np.zeros(self.n - start)))

        op.A = sp.vstack((op.A, A_upper_bounds[start:]))
        op.cType += 'U' * (self.n - start)
        if include_on_variables:
            op.b = np.hstack((op.b, np.zeros(self.n - start)))
        else:
            op.b = np.hstack((op.b, max_cap[start:]))
        # Minimum and maximum capacity for HEAT during start  -  an heat node given:
        if (shutdown_ramp_lower_bounds_heat is not None) and (self.idx_nodes['heat'] is not None):
            start = max(0, start_ramp_time - time_already_running) if time_already_running > 0 else 0
            A_lower_bounds = sp.lil_matrix((self.n, op.A.shape[1]))
            A_upper_bounds = sp.lil_matrix((self.n, op.A.shape[1]))
            starting_timestep = self.timegrid.restricted.I[0]
            for i in range(start, self.n):
                var = op.mapping.iloc[i]

                #A_lower_bounds[i, i] = 1
                A_lower_bounds[i, self.heat_idx + i] = 1 # conversion_factor_power_heat[i]  
                A_lower_bounds[i, self.on_idx + var["time_step"] - starting_timestep] = - 0

                #A_upper_bounds[i, i] = 1
                A_upper_bounds[i, self.heat_idx + i] = 1 # conversion_factor_power_heat[i] 
                A_upper_bounds[i, self.on_idx + var["time_step"] - starting_timestep] = - max_cap[i]/conversion_factor_power_heat[i] 

                for j in range(start_ramp_time):
                    if i - j < 0:
                        continue
                    A_lower_bounds[i, self.start_idx + i - j] = 0 -start_ramp_lower_bounds_heat[j]
                    A_upper_bounds[i, self.start_idx + i - j] = max_cap[i]/conversion_factor_power_heat[i] -start_ramp_upper_bounds_heat[j]

                for j in range(shutdown_ramp_time):
                    if i + j + 1 >= self.timegrid.restricted.T:
                        break
                    A_lower_bounds[i, self.shutdown_idx + i + j + 1] = 0 -shutdown_ramp_lower_bounds_heat[j]
                    A_upper_bounds[i, self.shutdown_idx + i + j + 1] = max_cap[i]/conversion_factor_power_heat[i] -shutdown_ramp_upper_bounds_heat[j]

            op.A = sp.vstack((op.A, A_lower_bounds[start:]))
            op.cType += 'L' * (self.n - start)
            op.b = np.hstack((op.b, np.zeros(self.n - start)))

            op.A = sp.vstack((op.A, A_upper_bounds[start:]))
            op.cType += 'U' * (self.n - start)
            if include_on_variables:
                op.b = np.hstack((op.b, np.zeros(self.n - start)))
            else:
                op.b = np.hstack((op.b, max_cap[start:]))


        # Enforce start_ramp if asset is in the starting process at time 0
        if time_already_running > 0 and time_already_running < start_ramp_time:
            for i in range(start_ramp_time - time_already_running):
                # Upper Bound:
                a = sp.lil_matrix((1, op.A.shape[1]))
                a[0, i] = 1
                if (self.idx_nodes['heat'] is not None): # heat note given        
                    a[0, self.heat_idx + i] = conversion_factor_power_heat[i]
                op.A = sp.vstack((op.A, a))
                op.cType += 'U'
                op.b = np.hstack((op.b, start_ramp_upper_bounds[time_already_running + i]))

                # Lower Bound:
                a = sp.lil_matrix((1, op.A.shape[1]))
                a[0, i] = 1
                if (self.idx_nodes['heat'] is not None): # heat note given        
                    a[0, self.heat_idx + i] = conversion_factor_power_heat[i]
                op.A = sp.vstack((op.A, a))
                op.cType += 'L'
                op.b = np.hstack((op.b, start_ramp_lower_bounds[time_already_running + i]))

        return op

    def _add_constraints_for_ramp(self, op: OptimProblem, ramp, conversion_factor_power_heat, 
                                  time_already_running, include_on_variables, max_cap, 
                                  start_ramp_time, shutdown_ramp_time, last_dispatch):
        """ Add ramp constraints to the OptimProblem op.
            These ensure that the increase/decrease of the virtual dispatch (power + conversion_factor_power_heat * heat)
            is bounded by ramp, except during timesteps that belong to the start or shutdown ramp """
        # Ramp constraints:
        if ramp is not None:
            for t in range(1, self.timegrid.restricted.T):
                # Lower Bound
                a = sp.lil_matrix((1, op.A.shape[1]))
                a[0, t] = 1
                if (self.idx_nodes['heat'] is not None): # heat note given        
                    a[0, self.heat_idx + t] = conversion_factor_power_heat[t]
                a[0, t - 1] = -1
                if (self.idx_nodes['heat'] is not None): # heat note given        
                    a[0, self.heat_idx + t - 1] = -conversion_factor_power_heat[t]
                if include_on_variables:
                    a[0, self.on_idx + t - 1] = ramp
                for i in range(shutdown_ramp_time):
                    if t + i >= self.timegrid.restricted.T:
                        break
                    a[0, self.shutdown_idx + t + i] = max_cap[t - 1] - ramp
                op.A = sp.vstack([op.A, a])
                op.cType += 'L'
                if include_on_variables:
                    op.b = np.hstack([op.b, 0])
                else:
                    op.b = np.hstack([op.b, -ramp])

                # Upper Bound
                a = sp.lil_matrix((1, op.A.shape[1]))
                a[0, t] = 1
                if (self.idx_nodes['heat'] is not None): # heat note given        
                    a[0, self.heat_idx + t] = conversion_factor_power_heat[t]
                a[0, t - 1] = -1
                if (self.idx_nodes['heat'] is not None): # heat note given        
                    a[0, self.heat_idx + t - 1] = -conversion_factor_power_heat[t]
                if include_on_variables:
                    a[0, self.on_idx + t] = -ramp
                    b_value = 0
                else:
                    b_value = ramp
                for i in range(start_ramp_time):
                    if t - i < 0:
                        if time_already_running > 0 and time_already_running - t + i == 0:
                            b_value += max_cap[t] - ramp
                            break
                        continue
                    a[0, self.start_idx + t - i] = ramp - max_cap[t]
                op.A = sp.vstack([op.A, a])
                op.cType += 'U'
                op.b = np.hstack([op.b, b_value])

            # Initial ramp constraint
            a = sp.lil_matrix((1, op.A.shape[1]))
            a[0, 0] = 1
            if (self.idx_nodes['heat'] is not None): # heat note given        
                a[0, self.heat_idx] = conversion_factor_power_heat[0]
            for i in range(shutdown_ramp_time):
                a[0, self.shutdown_idx + i] = last_dispatch - ramp
            op.A = sp.vstack([op.A, a])
            op.cType += 'L'
            if time_already_running == 0:
                op.b = np.hstack([op.b, last_dispatch])
            else:
                op.b = np.hstack([op.b, -ramp + last_dispatch])

            a = sp.lil_matrix((1, op.A.shape[1]))
            a[0, 0] = 1
            if (self.idx_nodes['heat'] is not None): # heat note given        
                a[0, self.heat_idx] = conversion_factor_power_heat[0]
            if include_on_variables:
                a[0, self.on_idx] = -ramp
            op.A = sp.vstack([op.A, a])
            op.cType += 'U'
            if not include_on_variables:
                op.b = np.hstack([op.b, last_dispatch + ramp])
            elif time_already_running > 0 and time_already_running > start_ramp_time:
                op.b = np.hstack([op.b, last_dispatch + max_cap[0] - ramp])
            else:
                op.b = np.hstack([op.b, last_dispatch])
        return op

    def _add_constrains_for_start_and_shutdown(self, op: OptimProblem, time_already_running, include_start_variables, include_shutdown_variables):
        """ Add constraints that ensure that the 'start' and 'shutdown' variables are correct """
        if include_start_variables:
            if not include_shutdown_variables:
                # Define just start constraints
                myA = sp.lil_matrix((self.timegrid.restricted.T - 1, op.A.shape[1]))
                for i in range(self.timegrid.restricted.T - 1):
                    myA[i, self.on_idx + i + 1] = 1
                    myA[i, self.on_idx + i] = - 1
                    myA[i, self.start_idx + i + 1] = -1
                op.A = sp.vstack((op.A, myA))
                op.cType += 'U' * (self.timegrid.restricted.T - 1)
                op.b = np.hstack((op.b, np.zeros(self.timegrid.restricted.T - 1)))

                if time_already_running == 0:
                    a = sp.lil_matrix((1, op.A.shape[1]))
                    a[0, self.on_idx] = 1
                    a[0, self.start_idx] = -1
                    op.A = sp.vstack((op.A, a))
                    op.cType += 'S'
                    op.b = np.hstack((op.b, 0))
            else:
                # Simultaneous definition of start- and shutdown constraints
                myA = sp.lil_matrix((self.timegrid.restricted.T - 1, op.A.shape[1]))
                for t in range(self.timegrid.restricted.T - 1):
                    myA[t, self.on_idx + t + 1] = 1
                    myA[t, self.on_idx + t] = - 1
                    myA[t, self.start_idx + t + 1] = -1
                    myA[t, self.shutdown_idx + t + 1] = 1
                op.A = sp.vstack((op.A, myA))
                op.cType += 'S' * (self.timegrid.restricted.T - 1)
                op.b = np.hstack((op.b, np.zeros(self.timegrid.restricted.T - 1)))

                if time_already_running == 0:
                    a = sp.lil_matrix((1, op.A.shape[1]))
                    a[0, self.on_idx] = 1
                    a[0, self.start_idx] = -1
                    op.A = sp.vstack((op.A, a))
                    op.cType += 'S'
                    op.b = np.hstack((op.b, 0))
                else:
                    a = sp.lil_matrix((1, op.A.shape[1]))
                    a[0, self.on_idx] = 1
                    a[0, self.shutdown_idx] = 1
                    op.A = sp.vstack((op.A, a))
                    op.cType += 'S'
                    op.b = np.hstack((op.b, 1))

                # Ensure that shutdown and start process do not overlap
                myA = sp.lil_matrix((self.timegrid.restricted.T - 1, op.A.shape[1]))
                for t in range(self.timegrid.restricted.T - 1):
                    myA[t, self.start_idx + t] = 1
                    myA[t, self.shutdown_idx + t] = 1
                op.A = sp.vstack((op.A, myA))
                op.cType += 'U' * (self.timegrid.restricted.T - 1)
                op.b = np.hstack((op.b, np.ones(self.timegrid.restricted.T - 1)))

                # Ensure that shutdown and start at timestep 0 are correct:
                if time_already_running == 0:
                    op.u[self.shutdown_idx] = 0
                else:
                    op.u[self.start_idx] = 0

        return op

    def _add_constraints_for_min_runtime(self, op: OptimProblem, min_runtime, include_start_variables, time_already_running):
        """ Add constraints to the OptimProblem op that ensure that every time the asset is turned on it remains on
            for at least the minimum runtime. """
        if include_start_variables and min_runtime > 1:
            for t in range(self.timegrid.restricted.T):
                for i in range(1, min_runtime):
                    if i > t:
                        continue
                    a = sp.lil_matrix((1, op.A.shape[1]))
                    a[0, self.on_idx + t] = 1
                    a[0, self.start_idx + t - i] = -1
                    op.A = sp.vstack((op.A, a))
                    op.cType += 'L'
                    op.b = np.hstack((op.b, 0))

            # Enforce minimum runtime if asset already on
            if time_already_running > 0 and min_runtime - time_already_running > 0:
                op.l[self.on_idx:self.on_idx + min_runtime - time_already_running] = 1
        return op

    def _add_constraints_for_min_downtime(self, op: OptimProblem, min_downtime, time_already_off):
        """ Add constraints to the OptimProblem op that ensure that every time the asset is turned off it remains off
            for at least the minimum downtime. """
        if min_downtime > 1:
            for t in range(self.timegrid.restricted.T):
                for i in range(1, min_downtime):
                    if i > t:
                        continue
                    a = sp.lil_matrix((1, op.A.shape[1]))
                    a[0, self.on_idx + t] = 1
                    a[0, self.on_idx + t - i] = -1
                    if t > i:
                        a[0, self.on_idx + t - i - 1] = 1
                    op.A = sp.vstack((op.A, a))
                    op.cType += 'U'
                    if not t > i and time_already_off == 0:
                        op.b = np.hstack((op.b, 0))
                    else:
                        op.b = np.hstack((op.b, 1))
            # Enforce minimum downtime if asset already off
            if time_already_off > 0 and min_downtime - time_already_off > 0:
                op.u[self.on_idx:self.on_idx + min_downtime - time_already_off] = 0
        return op

    def _add_constraints_for_heat(self, op: OptimProblem, max_share_heat):
        """ Add constraints to the OptimProblem op to bound the heat variable by max_share_heat * power. """
        # Boundaries for the heat variable:
        if max_share_heat is not None:
            myA = sp.lil_matrix((self.n, op.A.shape[1]))
            for i in range(self.n):
                myA[i, self.heat_idx + i] = 1
                myA[i, i] = - max_share_heat[i]
            op.A = sp.vstack((op.A, myA))
            op.cType += 'U' * self.n
            op.b = np.hstack((op.b, np.zeros(self.n)))
        return op

    def _add_fuel_consumption(self, op: OptimProblem, fuel_efficiency, consumption_if_on, start_fuel, conversion_factor_power_heat, include_on_variables, include_start_variables):
        """ In case there is an explicit node for fuel, extend the mapping.

            Idea: fuel consumption is  power disp + conversion_factor_power_heat * heat disp.
            To realise this, the mapping in the same way as in the simpler asset type 'MultiCommodityContract'."""
        # disp_factor determines the factor with which fuel is consumed
        if 'disp_factor' not in op.mapping: op.mapping['disp_factor'] = np.nan
        new_map = op.mapping.copy()
        if self.idx_nodes['heat'] is None: my_node_idx = [0] # do only for power
        else: my_node_idx = [0,1]  # do fo rheat and power node
        for i in my_node_idx:  # nodes power and heat
            initial_map = op.mapping[
                (op.mapping['var_name'] == 'disp') & (op.mapping['node'] == self.node_names[i])].copy()
            initial_map['node'] = self.node_names[self.idx_nodes['fuel']]  # fuel node
            if i == 0:
                initial_map['disp_factor'] = -1. / fuel_efficiency
            elif (i == 1):
                initial_map['disp_factor'] = -conversion_factor_power_heat / fuel_efficiency
            new_map = pd.concat([new_map, initial_map.copy()])
        # consumption  if on
        if include_on_variables:
            initial_map = op.mapping[op.mapping['var_name'] == 'bool_on'].copy()
            initial_map['node'] = self.node_names[self.idx_nodes['fuel']]  # fuel node
            # initial_map['var_name'] = 'fuel_if_on'
            initial_map['type'] = 'd'
            initial_map['disp_factor'] = -consumption_if_on
            new_map = pd.concat([new_map, initial_map.copy()])
        # consumption on start
        if include_start_variables:
            initial_map = op.mapping[op.mapping['var_name'] == 'bool_start'].copy()
            initial_map['node'] = self.node_names[self.idx_nodes['fuel']]  # fuel node
            # initial_map['var_name'] = 'fuel_start'
            initial_map['type'] = 'd'
            initial_map['disp_factor'] = -start_fuel
            new_map = pd.concat([new_map, initial_map.copy()])

        op.mapping = new_map
        return op

class CHPAsset_with_min_load_costs(CHPAsset):
    def __init__(self,
                 min_load_threshhold: Union[float, Sequence[float], StartEndValueDict] = 0.,
                 min_load_costs: Union[float, Sequence[float], StartEndValueDict, None] = None,
                **kwargs                 
                 ):
        """ CHPContract with additional Min Load costs: 
            adding costs when running below a threshhold capacity
        Args:

        CHPAsset arguments

        additional:

        min_load_threshhold (float: optional): capacity below which additional costs apply
        min_load_costs      (float: optional): costs that apply below a threshhold (fixed costs "is below * costs" independend of capacity)

        """
        super().__init__(**kwargs)
        self.min_load_threshhold = min_load_threshhold
        self.min_load_costs      = min_load_costs


    def setup_optim_problem(self, prices: dict, timegrid: Union[Timegrid, None] = None,
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

        min_load_threshhold = self.make_vector(self.min_load_threshhold, prices, default_value=0., convert=True)
        min_load_costs      = self.make_vector(self.min_load_costs, prices, default_value=0., convert = True)
        ### new part: add boolean "below threshhold" and restriction
        if (min_load_threshhold is not None) and (max(min_load_threshhold) >=0.)\
              and (min_load_costs is not None) and (max(min_load_costs) >=0.):

            ###  include bools:
            map_bool = pd.DataFrame()
            map_bool['time_step'] = self.timegrid.restricted.I
            map_bool['node'] = np.nan
            map_bool['asset'] = self.name
            map_bool['type'] = 'i'  # internal
            map_bool['bool'] = True
            map_bool['var_name'] = 'bool_threshhold'
            map_bool.index += op.mapping.index.max()+1 # those are new variables
            op.mapping = pd.concat([op.mapping, map_bool])
            # extend A for on variables (not relevant in exist. restrictions)
            op.A = sp.hstack((op.A, sp.lil_matrix((op.A.shape[0], len(map_bool)))))
            # set lower and upper bounds, costs:
            op.l = np.hstack((op.l, np.zeros(self.timegrid.restricted.T)))
            op.u = np.hstack((op.u, np.ones(self.timegrid.restricted.T)))
            op.c = np.hstack([op.c, min_load_costs])
            ### Define restriction
            node_power = self.nodes[0].name
            map_disp = op.mapping.loc[(op.mapping['node'] == node_power) & (op.mapping['var_name'] == 'disp'),:]
            map_bool = op.mapping.loc[(op.mapping['var_name'] == 'bool_threshhold'),:]
            map_bool_on = op.mapping.loc[(op.mapping['var_name'] == 'bool_on') & (op.mapping['node'].isnull()),:]
            assert len(map_disp)==len(map_bool), 'error- lengths of disp and bools do not match'
            # disp_t >= threshhold * (1-bool_t)  -  threshhold * (1- bool_on) 
            # disp_t + (bool_t - bool_on) * threshhold >= 0
            myA = sp.lil_matrix((len(map_disp), op.A.shape[1]))
            i_bool = 0 # counter booleans
            myb = np.zeros(len(map_disp))
            for t in map_disp['time_step'].values:
                ind_disp    = map_disp.index[map_disp['time_step'] == t][0]
                ind_bool    = map_bool.index[map_bool['time_step'] == t][0]
                myA[i_bool, ind_disp]    = 1
                myA[i_bool, ind_bool]    =  min_load_threshhold[t]
                if len(map_bool_on)>0:
                    ind_bool_on = map_bool_on.index[map_bool_on['time_step'] == t][0]
                    myA[i_bool, ind_bool_on] = -min_load_threshhold[t]
                    myb[i_bool]              = 0.
                else:
                    myb[i_bool]              = min_load_threshhold[t]
                i_bool += 1
            op.A = sp.vstack((op.A, myA))
            op.cType += 'L' * (len(map_disp))
            op.b = np.hstack((op.b, myb))

        return op

class Plant(CHPAsset):
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
                 **kwargs
                 ):
        """ Plant: Generate power (or another commodity) from fuel (fuel optional). Derived from more complex CHP, taking out heat
            Restrictions
            - time dependent capacity restrictions
            - MinTake & MaxTake for a list of periods
            - start costs
            - minimum runtime
            - ramps
        Args:
            name (str): Unique name of the asset                                              (asset parameter)
            nodes (Node): One node  for generated power                                       (asset parameter)
                          optional: node for fuel (e.g. gas)
            start (dt.datetime) : start of asset being active. defaults to none (-> timegrid start relevant)
            end (dt.datetime)   : end of asset being active. defaults to none (-> timegrid start relevant)
            timegrid (Timegrid): Timegrid for discretization                                  (asset parameter)
            wacc (float): Weighted average cost of capital to discount cash flows in target   (asset parameter)
            freq (str, optional):   Frequency for optimization - in case different from portfolio (defaults to None, using portfolio's freq)
                                    The more granular frequency of portf & asset is used
            profile (pd.Series, optional):  If freq(asset) > freq(portf) assuming this profile for granular dispatch (e.g. scaling hourly profile to week).
                                            Defaults to None, only relevant if freq is not none
            min_cap (float) : Minimum capacity for generating virtual dispatch (power + conversion_factor_power_heat * heat). Has to be greater or equal to 0. Defaults to 0.
            max_cap (float) : Maximum capacity for generating virtual dispatch (power + conversion_factor_power_heat * heat). Has to be greater or equal to 0. Defaults to 0.
            min_take (float) : Minimum volume within given period. Defaults to None
            max_take (float) : Maximum volume within given period. Defaults to None
                              float: constant value
                              dict:  dict['start'] = np.array
                                     dict['end']   = np.array
                                     dict['values"] = np.array
            price (str): Name of price vector for buying / selling
            extra_costs (float, dict, str): extra costs added to price vector (in or out). Defaults to 0.
                                            float: constant value
                                            dict:  dict['start'] = array
                                                   dict['end']   = array
                                                   dict['values"] = array
                                            str:   refers to column in "prices" data that provides time series to set up OptimProblem (as for "price" below)
            periodicity (str, pd freq style): Makes assets behave periodicly with given frequency. Periods are repeated up to freq intervals (defaults to None)
            periodicity_duration (str, pd freq style): Intervals in which periods repeat (e.g. repeat days ofer whole weeks)  (defaults to None)
            ramp (float): Maximum increase/decrease of virtual dispatch (power + conversion_factor_power_heat * heat) in one main time unit. Defaults to None.
            start_costs (float): Costs for starting. Defaults to 0.
            running_costs (float): Costs when on. Defaults to 0.
            min_runtime (int): Minimum runtime in timegrids main_time_unit. (start ramp time and shutdown ramp time do not count towards the min runtime.) Defaults to 0.
            time_already_running (int): The number of timesteps the asset is already running in timegrids main_time_unit. Defaults to 0.
            min_downtime (int): Minimum downtime in timegrids main_time_unit. Defaults to 0.
            time_already_off (int): The number of timesteps the asset has already been off in timegrids main_time_unit. Defaults to 0.
            last_dispatch (float): Previous virtual dispatch (power + conversion_factor_power_heat * heat). Defaults to 0.
            start_ramp_lower_bounds (Sequence): The i-th element of this sequence specifies a lower bound of the
                                                virtual dispatch (power + conversion_factor_power_heat * heat) at i timesteps
                                                of freq ramp_freq after starting.  Defaults to None.
            start_ramp_upper_bounds (Sequence): The i-th element of this sequence specifies an upper bound of the
                                                virtual dispatch (power + conversion_factor_power_heat * heat) at i timesteps
                                                of freq ramp_freq after starting.  Defaults to None.
            shutdown_ramp_lower_bounds (Sequence): The i-th element of this sequence specifies a lower bound of the
                                                   virtual dispatch (power + conversion_factor_power_heat * heat) at i timesteps
                                                   of freq ramp_freq before turning off. Defaults to None.
            shutdown_ramp_upper_bounds (Sequence): The i-th element of this sequence specifies an upper bound of the
                                                   virtual dispatch (power + conversion_factor_power_heat * heat) at i timesteps
                                                   of freq ramp_freq before turning off. If it is None, it is set equal to shutdown_ramp_upper_bounds.
                                                   Defaults to None.
            ramp_freq (str): A string specifying the frequency of the start-and shutdown ramp specification.
                             If this is None, the timegrids main_time_unit is used. Otherwise the start and shutdown ramps are
                             interpolated to get values in the timegrids freq.


            Optional: Explicit fuel consumption (e.g. gas) for multi-commodity simulation
                 start_fuel (float, dict, str): detaults to  0
                 fuel_efficiency (float, dict, str): defaults to 1
                 consumption_if_on (float, dict, str): defaults to 0
        """
        super(Plant, self).__init__(name=name,
                                       nodes=nodes,
                                       start=start,
                                       end=end,
                                       wacc=wacc,
                                       freq=freq,
                                       profile=profile,
                                       price=price,
                                       extra_costs=extra_costs,
                                       min_cap=min_cap,
                                       max_cap=max_cap,
                                       min_take=min_take,
                                       max_take=max_take,
                                       periodicity=periodicity,
                                       periodicity_duration=periodicity_duration,
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

class CHP_PQ_diagram(CHPAsset): 
    def __init__(self,
                 pq_polygon: Union[List[Union[List[float], np.ndarray]], None] = None,
                 **kwargs                 
                 ):
        """ CHPContract using a convex polygon to define feasible (P,Q) operating points: 

        Args:
        * CHPAsset arguments

        additional:
        pq_polygon (list of 2-element lists or arrays): 2D points [P, Q] in convex polygon - given as lists or arrays of 2 elements
                                                        e.g. [[0,0], [1,0], [1,1], [0,1]] for a square
                                                        Order of points is relevant! Polygon has to be convex
        """
        super().__init__(**kwargs)
        # do some checks and store polygon
        if pq_polygon is None:  
            print('Warning: No PQ diagram polygon given. Use CHPAsset if no polygon is needed')
        else:
            if len(pq_polygon) < 3: raise ValueError('Error - PQ diagram polygon has to have at least 3 points')
            check = self._check_polygon(pq_polygon)
            if check == 0: raise ValueError('Error - PQ diagram polygon is not convex')
            # do we have max_share_heat given? That's compatiple, but not needed. Warn user
            if self.max_share_heat is not None:
                print('Warning: max_share_heat given, but not needed when using PQ diagram. Think about removing it.')
            # heat node given?
            if self.idx_nodes['heat'] is None:
                raise ValueError('Error - no heat node given, but pq_polygon given. Use Plant asset?')

        self.pq_polygon = pq_polygon # add anyhon - if None OptimProblem of CHPAsset is used

    @staticmethod
    def _check_polygon(points:List[Union[List[float], np.ndarray]]) -> int:
        """ Check validity and orientation of polygon (is convex?)
            How: calculate cross product of each edge with the next edge. 
                 All positive: clock, all negative: counterclockwise, mixed: not convex
            Args:
                points (dict): 2D points in polygon - given as lists or arrays of 2 elements
            Returns:
                int: 0: not valid, 1: valid clockwise, -1: valid counterclockwise
        """
        def cross_product(a, b, c):
            return ((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))  # z component of cross product, edges a - b - c
      
        prev = 0
        n = len(points)
        for i in range(n):
            cp = -cross_product(points[i], 
                                points[(i + 1) % n], 
                                points[(i + 2) % n])
            if cp != 0:
                if prev == 0: prev = cp
                elif cp * prev < 0:
                    return 0 # not convex
        return np.sign(cp)
    
    def setup_optim_problem(self, prices: dict, timegrid: Union[Timegrid, None] = None,
                            costs_only: bool = False) -> OptimProblem:
        """ Set up optimization problem for asset. Use super class and add polygon restriction on heat & power

        Args:
            prices (dict): Dictionary of price arrays needed by assets in portfolio
            timegrid (Timegrid, optional): Discretization grid for asset. Defaults to None,
                                           in which case it must have been set previously
            costs_only (bool): Only create costs vector (speed up e.g. for sampling prices). Defaults to False

        Returns:
            OptimProblem: Optimization problem to be used by optimizer
        """
        op = super().setup_optim_problem(prices=prices, timegrid=timegrid, costs_only=costs_only)
        if self.pq_polygon is None:
            return op    # return as is - no polygon given
        poly_type = self._check_polygon(self.pq_polygon)
        assert poly_type != 0, 'Implementation error - polygon not convex, should have been caught before'
        # add polygon restriction
        n_poly = len(self.pq_polygon)
        # get node names and indices of dispatch variables
        n_power   = self.node_names[self.idx_nodes['power']]
        n_heat    = self.node_names[self.idx_nodes['heat']]
        ind_power = op.mapping.loc[(op.mapping['node'] == n_power) & (op.mapping['type'] == 'd')].index
        ind_heat  = op.mapping.loc[(op.mapping['node'] == n_heat)  & (op.mapping['type'] == 'd')].index
        assert len(ind_power) == len(ind_heat) == self.timegrid.restricted.T, 'Implementation error - number of dispatch variables not correct'
        n = len(ind_power) 
        m = op.A.shape[1]-2*n
        assert op.A.shape[1]-2*n >=0, 'Implementation error - number of variables not correct'

        for i in range(n_poly):  # loop over all edges
            # looking at the restriction given by edge p(i) and p(i+1)
            a = self.pq_polygon[i]
            b = self.pq_polygon[(i+1) % n_poly]
            if (b[1]-a[1]) == 0:     # vertical in PQ, Q constant
                #### effectively new minimum or maximum heat
                if      ((a[0] < b[0]) and (poly_type == -1)) \
                    or  ((a[0] > b[0]) and (poly_type ==  1)) :  # increase minimum heat
                    op.l[ind_heat] = np.maximum(op.l[ind_heat], np.ones(n)*a[1])
                else:           #  reduce maximum heat
                    op.u[ind_heat] = np.minimum(op.u[ind_heat], np.ones(n)*a[1])
            elif (b[0]-a[0]) == 0:   # horizontal in PQ, P constant)
                    #### effectively new minimum or maximum power
                    if      ((a[1] < b[1]) and (poly_type == -1)) \
                        or  ((a[1] > b[1]) and (poly_type ==  1)) :  # decrease maximum power
                        op.u[ind_power] = np.minimum(op.u[ind_power], np.ones(n)*a[0])
                    else:           #  increase minimum power
                        op.l[ind_power] = np.maximum(op.l[ind_power], np.ones(n)*a[0])
            else:  # determine line equation:  P = m_eq * Q + b_eq    
                m_eq = (b[0]-a[0])/(b[1]-a[1])
                b_eq = a[0] - m_eq*a[1]  # b = P1-mQ1
                #### extend restrictions
                myA = sp.hstack([sp.eye(n), -m_eq*sp.eye(n), sp.lil_matrix((n, m))])
                myb = np.ones(n)*b_eq
                if      ((a[1] < b[1]) and (poly_type == -1)) \
                    or  ((a[1] > b[1]) and (poly_type ==  1)) :
                    mytype = 'U'*n 
                else:
                    mytype = 'L'*n 
                #### stack
                op.A      = sp.vstack((op.A, myA))
                op.cType +=  mytype
                op.b = np.hstack((op.b, myb))
        return op
    