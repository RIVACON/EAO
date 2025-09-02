import numpy as np
import pandas as pd
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '../../../..'))
import eaopack as eao

# define file names for this sample
## input data
sample_file   = join(mypath, 'portfolio_simple_start.json')
timegrid_file = join(mypath, 'timegrid.json')
## output files for results
results_file  = join(mypath, 'results_simple_start.xlsx')
graph_file    = join(mypath, 'network_graph_simple_start.pdf')

###############################################   load data 
print('load data')
portf = eao.serialization.load_from_json(file_name= sample_file)
timegrid = eao.serialization.load_from_json(file_name=timegrid_file)
prices = {'price sourcing': 5.*np.sin(np.linspace(0.,6., timegrid.T)),
         'price sales a' : np.ones(timegrid.T)*2.,
         'price sales b' : 5.+5.*np.sin(np.linspace(0.,6., timegrid.T))  }

###############################################   visualization
print('create and write network graph to pdf file')
eao.network_graphs.create_graph(portf = portf, file_name= graph_file, title = 'Simple portfolio example')
# alternatively show graph:
# eao.network_graphs.create_graph(portf = portf, file_name= None)

###############################################   optimization
print('run portfolio optimization')
optim_problem  = portf.setup_optim_problem(prices, timegrid)
result         = optim_problem.optimize()

###############################################   extracting results
print('extract results and write to file')
output = eao.io.extract_output(portf, optim_problem, result, prices)
eao.io.output_to_file(output=output, file_name= results_file)
