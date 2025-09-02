### capturesample portfolio and write to JSON


import pandas as pd
import datetime as dt
import numpy as np

import os, sys
from os import path as ospath

from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '..'))

from pandas.core.indexes.base import ensure_index

## in case paths have not been set correctly
myDir = ospath.abspath(ospath.join(os.path.dirname(__file__),"../../../.."))
sys.path.insert(0, myDir)

import eaopack as eao

def capture_asset():
    ################################### define parameters
    asset_file = os.path.join(ospath.join(os.path.dirname(__file__)),'heat_sample.JSON')

    ################################## import data and set up portfolio

    ## nodes
    unit_power = eao.assets.Unit(volume = 'MWh(el)', flow = 'MW(el)')
    node_power = eao.assets.Node(name = 'power', unit = unit_power)
    unit_heat  = eao.assets.Unit(volume = 'MWh(th)', flow = 'MW(th)')
    node_heat     = eao.assets.Node(name = 'heat', unit = unit_heat)

    ## create battery
    storage = eao.assets.Storage(name       = 'heat_storage', 
                                nodes       = node_heat, 
                                cap_out     = 1., 
                                cap_in      = 1., 
                                size        = 10., 
                                start_level = 0., 
                                end_level   = 0.,
                                eff_in      = 0.9, 
                                block_size  = 'd')     # optimization for each day independently 

    power2heat   = eao.assets.Transport(name = 'power-2-heat', efficiency = 4., nodes = [node_power, node_heat])
    power_gen    = eao.assets.MultiCommodityContract(name = 'CHP', extra_costs = 50, min_cap= 0, max_cap=1, nodes = [node_power, node_heat], factors_commodities=[0.8, 2.2])
    market   = eao.assets.SimpleContract(name = 'power_market', price='price', min_cap= -10, max_cap=10, nodes = node_power)

    Start = dt.date(2020,1,1)
    End   = dt.date(2021,1,1)
    dates = pd.date_range(Start, End, freq = 'd').values

    heat_curve = {'start'  : dates, 
                 'values' : - np.sin(np.linspace(0,10, len(dates)))}
    heat_demand  = eao.assets.Contract(name = 'heat_demand',  extra_costs = -20, min_cap= heat_curve, max_cap=heat_curve, nodes = node_heat)
    portf        = eao.portfolio.Portfolio([storage, power2heat, power_gen, market, heat_demand])
    portf.set_timegrid(eao.assets.Timegrid(dt.date(2020,1,1), dt.date(2020,2,1), freq = 'h'))
    ## write to JSON
    eao.serialization.to_json(portf, file_name=asset_file)

################# standalone
if __name__ == "__main__" :
    capture_asset()
