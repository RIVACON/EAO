import unittest
import numpy as np
import datetime as dt
import time
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '..'))

import eaopack as eao

class Solvers(unittest.TestCase):
    def test_solverCPLEX(self):
        """ Unit test. Setting up a simple contract with random prices 
            and check that it buys full load at negative prices and opposite
        """
        ### test inactive if CPLEX not installed
        try:
            import cplex
        except: pass
        else:
            node = eao.assets.Node('testNode')
            timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,2,1), freq = 'd')
            a = eao.assets.SimpleContract(name = 'SC', price = 'rand_price', nodes = node ,
                            min_cap= -10., max_cap=10.)
            #a.set_timegrid(timegrid)
            prices ={'rand_price': np.random.rand(timegrid.T)-0.5}
            op = a.setup_optim_problem(prices, timegrid=timegrid)
            res = op.optimize(solver='CPLEX')
            x = np.around(res.x, decimals = 3) # round
            check =     all(x[np.sign(op.c) == -1] == op.u[np.sign(op.c) == -1]) \
                    and all(x[np.sign(op.c) == 1]  == op.l[np.sign(op.c) == 1])
            tot_dcf = np.around((a.dcf(op, res)).sum(), decimals = 3) # asset dcf, calculated independently
            check = check and (tot_dcf == np.around(res.value , decimals = 3))
            self.assertTrue(check)


    def test_MIP_CPLEX(self):
        ### test inactive if CPLEX not installed
        try:
            import cplex
        except: pass
        else:
            node1 = eao.assets.Node('N1')
            node2 = eao.assets.Node('N2')
            timegrid = eao.assets.Timegrid(dt.date(2021,1,1), dt.date(2021,1,5), freq = '3h')
            a = eao.assets.Storage('a1', [node1, node2], start=dt.datetime(2020,12,31,23), end=dt.date(2021,2,1),size=10,\
                cap_in=1, cap_out=1, start_level=0, end_level=0, cost_in=.1, eff_in= 0.7,
                no_simult_in_out=True, max_store_duration= 7)
            buy  = eao.assets.SimpleContract(name = 'buy', nodes = node1, min_cap=-10, max_cap=10, price = 'price')
            sell = eao.assets.SimpleContract(name = 'sell', nodes = node2, min_cap=-10, max_cap=10, price = 'price')        
            # cannot be used, since storage can only BRING volumes from node1 to node2
            portf = eao.portfolio.Portfolio([a, buy, sell])
            test_string = eao.serialization.json_serialize_objects(portf)
            price = -np.sin(np.linspace(0,12,timegrid.T))
            prices ={ 'price': price, 'zero':np.zeros(timegrid.T), 'best':-np.ones(timegrid.T)}
            op = portf.setup_optim_problem(prices, timegrid=timegrid)

            ### standard solver
            print('std. solver')
            s1 = time.perf_counter()
            res1 = op.optimize()
            print('Time: '+ str(time.perf_counter()-s1))
            ### cplex
            print('CPLEX')
            s1 = time.perf_counter()
            res2 = op.optimize(solver = 'CPLEX')
            print('Time: '+ str(time.perf_counter()-s1))
            self.assertAlmostEqual(res2.value, res1.value, 3)



        
###########################################################################################################
###########################################################################################################
###########################################################################################################

if __name__ == '__main__':
    unittest.main()
