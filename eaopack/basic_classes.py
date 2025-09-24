from typing import Union, List, Dict, TypedDict, Sequence
import datetime as dt
import numpy as np
import pandas as pd

class StartEndValueDict(TypedDict):
    """ New type to contain info of the type start, end, value
        dict with
            start: array of datetime 
            end:   array of datetime (optional)
            values: array of floats """
    start:  Sequence[dt.datetime]
    end:    Sequence[dt.datetime]
    values: Sequence[float]

class Unit:
    def __init__(self, volume:str='MWh', flow:str='MW', factor:float=1.):
        """ Defines the units used in a node and how to convert between volums and flows (volume/time)

        Args:
            volume (str, optional):'volume' is a quantity unit, typically MWh or MJ for energy. Defaults to 'MWh'.
            flow (str, optional): 'flow' is a quantity per time unit, typically MW (J/s) for energy. Defaults to 'MW'.
            factor (float, optional): 'factor' is measured in the <main time unit> of the optimization problem (default 'h'). 
                                       It is the value of (volume/flow). Defaults to 1 <basic time unit>.
                                       The factor for conversions may prove very handy, particularly with multi commodity models
                                       however, it is not implemented yet. We included the factor to ensure at the start that 
                                       only unit valid choices with factor 1 are used. Conversion to be added in later version 
        """
        self.volume = volume
        self.flow   = flow
        self.factor = factor
        assert (factor == 1.), 'conversion of volume/flow units using factor not implemented. please choose units with conversion factor 1'

class Node:
    def __init__(self, name: str, commodity: Union[None, str] = None, unit: Unit = Unit()) :
        """ Class to define a node in the optimization problem. A node is a (virtual) point, where
            assets are located. In a node, the sum of all commodity flows must be zero.
            Only one commodity may be present in each node. 
            Per node we also define the units to be used for capacity (volume/energy and flow/capacity).
            Examples are MWh and MW or liters and liters per minute.

        Args:
            name (str): Name of the node (must be unique in the portfolio)
            commodity (str): Commodity traded in the node. None possible if there is no distinction. Defaults to None
            unit (Unit): Units used in the node. Defaults to default unit     
        """
        self.name = name
        self.commodity = commodity
        self.unit = unit

class Timegrid:
    def __init__(self,  start:          Union[None, dt.date, dt.datetime, pd.Timestamp],
                        end:            Union[None, dt.date, dt.datetime, pd.Timestamp],
                        freq:           str = 'h', 
                        main_time_unit: str = 'h', 
                        ref_timegrid     = None, 
                        timezone:       Union[None, str]      = None):
        """ Manage the timegrid used for optimization. 

        Args:
            start (dt.datetime): Start datetime
            end (dt.datetime): End datetime
            freq (str, optional): Frequency for discretization according to pandas notation ('15min', 'h', 'd', ...). Defaults to 'h'
            main_time_unit (str, optional): All times in the optimization problem are measured in the main_time_unit. Pandas notation. Defaults to 'h'
            timezone: Timezone for times. String according to pandas tz definitions (e.g. CET). Defaults to None (naive timezone)
            ref_timegrid (Timegrid, optional): reference TG in case this timegrid is a subset of a suber grid
        Returns:
            timepoints (daterange): specific points in time of mesh
            T (float): number of time steps
            dt (np.array): time step for each step
            Dt (np.array): cummulative time steps (total duration since start)
        """
        self.freq = freq
        self.main_time_unit = main_time_unit

        # some timezone specific checks and definitions
        # also converting dates to pd.Timestamp to simplify further coding / avoid mismatches
        if ref_timegrid is not None:
            # get timezone from reference tg
            self.tz = ref_timegrid.tz
        else:
            self.tz = timezone

        # convert to timestamp to avoid dt/pd confusion and allow all inputs covered by pd.Timestamp
        if not start is None:
            self.start = pd.Timestamp(start)
        if not end is None:            
            self.end   = pd.Timestamp(end)

        # use reference TG and ignore start/end - then filter
        if ref_timegrid is not None:
            if self.start is None:
                self.start = ref_timegrid.start
            else:
                if self.start.tzinfo is None:
                    self.start = pd.Timestamp(self.start, tz = self.tz)
                else:
                    self.start = pd.Timestamp(self.start)          
            if self.end is None:
                self.end = ref_timegrid.end   
            else:
                if self.end.tzinfo is None:
                    self.end = pd.Timestamp(self.end, tz = self.tz)
                else:
                    self.end = pd.Timestamp(self.end)          
            # restricted timegrid is generated by filtering the main timegrid
            if freq == ref_timegrid.freq:
                I = (ref_timegrid.timepoints>=self.start) & (ref_timegrid.timepoints<self.end)
                self.I = ref_timegrid.I[I]
                self.timepoints = ref_timegrid.timepoints[I]
                self.dt = ref_timegrid.dt[I]
                self.Dt = ref_timegrid.Dt[I]
                self.T = len(self.timepoints)
                if hasattr(ref_timegrid,'discount_factors'):
                    self.discount_factors = ref_timegrid.discount_factors[I]
                # if  frequency is different from reference do further filtering
            else:
                try:
                    freq_a = pd.Timedelta(1, freq)
                except:
                    freq_a = pd.Timedelta(freq)
                try:
                    freq_p = pd.Timedelta(1, ref_timegrid.freq)
                except:
                    freq_p = pd.Timedelta(ref_timegrid.freq)                    
                assert (freq_a >= freq_p), 'timegrid must have less/equal granular frequency than reference'
                # generate more granular timegrid - all minor grid points in "right interval"  |->|->|->|
                pts  = (pd.date_range(start=self.start, end = self.end, freq = freq, tz = self.tz))
                self.I          = []
                self.I_minor_in_major = [] # to contain the list of minor grid points in major grid
                self.timepoints = []
                self.dt         = []
                self.Dt         = []                
                if hasattr(ref_timegrid,'discount_factors'):
                    self.discount_factors = []
                for a,b in zip(pts[0:-1], pts[1:]):
                    I = (ref_timegrid.timepoints >= a) & (ref_timegrid.timepoints < b)
                    self.I_minor_in_major.append(ref_timegrid.I[I])
                    myI = ref_timegrid.I[I].min() # last minor grid in major grid
                    self.I.append(myI)
                    self.dt.append(ref_timegrid.dt[I].sum())
                    self.Dt.append(ref_timegrid.Dt[myI])
                    self.timepoints.append(ref_timegrid.timepoints[myI])
                    if hasattr(ref_timegrid,'discount_factors'):
                        self.discount_factors.append(ref_timegrid.discount_factors[myI])      
                self.I          = np.asarray(self.I)
                self.timepoints = np.asarray(self.timepoints)
                self.dt         = np.asarray(self.dt)
                self.Dt         = np.asarray(self.Dt)
                self.T          = len(self.timepoints)
                if hasattr(ref_timegrid,'discount_factors'):
                    self.discount_factors = np.asarray(self.discount_factors)
                pass
        # no reference TG given
        else:
            if self.start.tzinfo is None:
                self.start = pd.Timestamp(self.start, tz = self.tz)
            if self.end.tzinfo is None:
                self.end = pd.Timestamp(self.end, tz = self.tz)
            assert self.start < self.end
            timepoints      = pd.date_range(start=self.start, end = self.end, freq = self.freq, tz = self.tz)
            self.timepoints = timepoints[0:-1] # specific time points of mesh
            self.dt         = (timepoints[1:]-timepoints[0:-1])/pd.Timedelta(1, self.main_time_unit) # time steps in mesh 
            self.Dt         = np.cumsum(self.dt) # total duration since start (in main time unit)
            self.T          = len(self.timepoints) #  number of time steps in mesh
            self.I          = np.array(range(0,self.T))

            self.dt = self.dt.values
            self.Dt = self.Dt.values

    def set_wacc(self, wacc:float):
        """ use wacc to create discount factors for discounted cash flows
        Args:
            wacc (float): weighted average cost of capital
        """
        # compute corresponding discount factors
        d = (1.+wacc)**(1./365.) # convert interest rate to daily
        self.discount_factors =  1./d**(self.Dt*pd.Timedelta(1, self.main_time_unit)/pd.Timedelta(1, 'd')) 

    def prices_to_grid(self, prices: dict):
        """
        Convert prices into a dataframe with index self.timepoints.
        If the price points already have (possibly different) timepoints, then the prices at self.timepoints
        are interpolated.

        Args:
            prices: This can be either a pandas dataframe with a Datetimeindex where each column corresponds to a price,
                or a dictionary of prices. In the case of a dict, the items can have the following forms:
                - array of length self.T. In this case it is assumed that the i-th entry of the array corresponds to the
                    i-th point in self.timepoints.
                - dict where the keys are timepoints and the corresponding items depict the prices at the specific
                    timepoints

        Returns:
            prices:
        """
        if not isinstance(prices, pd.DataFrame):
            prices = pd.DataFrame.from_dict(prices)
        if not isinstance(prices.index, pd.DatetimeIndex):
            if pd.api.types.is_any_real_numeric_dtype(prices.index):
                prices.index = self.timepoints
            else:
                prices.index = pd.to_datetime(prices.index)
        # check: time grid covered by data?
        if prices.index.values[0]>self.timepoints[0]:
            raise ValueError("Error: Data provided starts later than timegrid -- check input data")
        if prices.index.values[-1]<self.timepoints[-1]:
            raise ValueError("Error: Data provided ends before timegrid -- check input data")        
        prices = prices.reindex(prices.index.union(self.timepoints))
        prices = prices.interpolate(method='time', limit_direction="both")
        prices = prices.loc[self.timepoints]
        return prices

    def set_restricted_grid(self,start:dt.datetime = None, end:dt.datetime = None, freq:bool = None):
        """ return dictionary of arrays restricted to start/end
            used typically for assets valid in restricted timeframe
        Args:
            start, end (dt.datetime): start and end of restricted arrays. Default to None --> full timegrid's start/end
            freq (bool, optional):    frequency for restr. timegrid. may be chosen less granular than tg. Defaults to None
        """
        # if start/end not given, use those from timegrid
        if start is None: start = self.start
        if end   is None: end   = self.end
        if freq is None: # standard case
            self.restricted = Timegrid(start, end, freq=self.freq, main_time_unit=self.main_time_unit, ref_timegrid = self)
        else: # in case the restricted timegrid should have a different frequency:
            self.restricted = Timegrid(start, end, freq=freq, main_time_unit=self.main_time_unit, ref_timegrid = self)

    def prep_date_dict(self, dd:dict):
        """ Using dicts with keys "start", "end" and "values" throughout. Could change to an own classe
            but so far sticking to this definition. This function converts the dict e.g. by changing to timegrid
            time zone

        Args:
            dd (dict): date dict -- dict with keys "start', "end", "values"

        Returns:
            date dict with same keys
        """
        assert ('values' in dd)
        dd = dd.copy()
        out = {}
        keys = ['start', 'end']
        for myk in keys:
            out[myk] = []
            if myk in dd:
                for v in dd[myk]:
                    if not isinstance(v, pd.Timestamp):
                        v = pd.Timestamp(v)
                    if v.tzinfo is None:
                        v = v.tz_localize(self.tz)
                    out[myk].append(v)
        out['values'] = dd['values']
        return out

    def values_to_grid(self,inp:StartEndValueDict) -> np.array:
        """ Assignment of data from interval data to timegrid
            Args:
                inp (dict) with keys --- start, end, values
                                         each defining time interval where value is to be used

            Returns:
                array of values on the time grid
         """
        assert ('start' in inp)
        if isinstance(inp['start'],pd.DatetimeIndex):
            inp['start'] = inp['start'].tolist()
        if not isinstance(inp['start'], (list, np.ndarray)):
            inp['start'] = [inp['start']]
        if 'end' in inp:
            if isinstance(inp['end'],pd.DatetimeIndex):
                inp['end'] = inp['end'].tolist()            
            if not isinstance(inp['end'], (list, np.ndarray)):
                inp['end'] = [inp['end']]
        assert ('values' in inp)
        if not isinstance(inp['values'], (list, np.ndarray)):
                inp['values'] = [inp['values']]                
        # two cases: (1) "end" given or (2) "end" not given and implicitly start of the next interval
        if not 'end' in inp:
            if len(inp['start']) > 1: 
                inp['end']       = inp['start'].copy() # initialize, unknown type
                inp['end'][:-1]  = inp['start'][1:]
                inp['end'][-1]   = inp['end'][-1] + 2*(inp['start'][-1]-inp['start'][-2]) # generously extend validity
            else: # only one start value given, valid "for ever"
                inp['end'] = [pd.Timestamp.max]
        grid = np.empty(self.T)
        grid[:] = np.nan
        # catch and heal potential timezone mismatch in input data
        if (pd.to_datetime(inp['start']).tz is None) and (self.tz is not None):
            inp['start'] = pd.to_datetime(inp['start']).tz_localize(self.tz)
        if (pd.to_datetime(inp['end']).tz is None) and (self.tz is not None):
            inp['end']   = pd.to_datetime(inp['end']).tz_localize(self.tz)
        for s, e, v in zip(inp['start'], inp['end'], inp['values']):
            I = (self.timepoints >= pd.to_datetime(s)) & (self.timepoints < pd.to_datetime(e))
            if not all(np.isnan(grid[I])):
                raise ValueError('Overlapping time intervals')
            grid[I] = v
        return grid

def convert_time_unit(value: float, old_freq:str, new_freq:str) -> float:
    """
    Convert time value from old_freq to new_freq
    Args:
        value (float): the time value to convert
        old_freq: pandas frequency string, e.g. 'd', 'h', 'min', '15min', '1d1h'
        new_freq: pandas frequency string, e.g. 'd', 'h', 'min', '15min', '1d1h'

    Returns:
        the time value converted from old_freq to new_freq
    """
    return value * pd.to_timedelta(pd.tseries.frequencies.to_offset(old_freq)) / pd.to_timedelta(pd.tseries.frequencies.to_offset(new_freq))