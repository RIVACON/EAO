import numpy as np
import pandas as pd
import datetime as dt

# in case eao is not installed
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '../../../..'))
import eaopack as eao

import eaopack
from eaopack.assets import Node, Timegrid, Contract, Storage, SimpleContract
from eaopack.portfolio import Portfolio


# define file names for this sample
## input data
sample_file   = join(mypath, 'portfolio_renewable_storage.json')
timegrid_file = join(mypath, 'timegrid.json')
## output files for results
results_file  = join(mypath, 'portfolio_renewable_storage.xlsx')
graph_file    = join(mypath, 'network_graph_portfolio_renewable_storage.pdf')

###############################################   create portfolio 
print('create portfolio data')

Start = dt.date(2021,1,1)
End   = dt.date(2021,1,10)
timegrid = Timegrid(Start, End, freq = 'h')
node  = Node(name = 'node', commodity='power', unit='MWh')

prices = {'spot_market': 50+12.*(np.sin(np.linspace(0.,100., timegrid.T)))}

print('Portfolio ingredients:')
print('...spot market with basically unlimited possibility to buy and sell')
spot_market = SimpleContract(name = 'spot market', price = 'spot_market', min_cap = -1000, max_cap = +1000, nodes= node)
print('...a PV asset with typical (fixed) daily profile ')
PV_gen = {'start': timegrid.timepoints.to_list(),
          'values':abs(np.sin(np.pi/24 * timegrid.timepoints.hour.values))}
PV         = Contract(name = 'PV', min_cap= 0., max_cap= PV_gen, nodes = node)
print('...a wind asset with typical (fixed) daily profile ')
wind_gen = {'start': timegrid.timepoints.to_list(),\
            'values': 5.*np.random.rand(timegrid.T)}
wind        = Contract(name = 'wind', min_cap= 0., max_cap= wind_gen, nodes = node)
print('...a load profile to be delivered')
load_profile = {'start': timegrid.timepoints.to_list(),\
                'values': -5*abs(np.sin(np.pi/22 * timegrid.timepoints.hour.values)     \
                              + np.sin(0.1 + np.pi/10 * timegrid.timepoints.hour.values)\
                              + np.sin(0.2 + np.pi/3 * timegrid.timepoints.hour.values) )}
load        = Contract(name = 'load', min_cap= load_profile, max_cap= load_profile, nodes = node)

print('...a battery storage with 90% cycle efficiency')
storage = Storage(name = 'battery', cap_in= 1, cap_out=1, size = 4, \
                  eff_in=0.9, nodes = node, cost_in=1, cost_out=1)

portf = Portfolio([spot_market, PV, wind, load, storage])
###############################################   visualization
print('create and write network graph to pdf file')
eao.network_graphs.create_graph(portf = portf, file_name= graph_file, title = 'Sample for portfolio of RES with storage')
# alternatively show graph:
# eao.network_graphs.create_graph(portf = portf, file_name= None)

###############################################   optimization
print('run portfolio optimization')
optim_problem  = portf.setup_optim_problem(prices, timegrid)
result         = optim_problem.optimize()

###############################################   extracting results
print('extract results and write to file')
output = eao.io.extract_output(portf, optim_problem, result, prices)
writer = pd.ExcelWriter(results_file)
for myk in output:
        output[myk].to_excel(writer, sheet_name = myk)
writer.close()
