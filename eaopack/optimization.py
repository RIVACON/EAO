import numpy as np
import pandas as pd
import datetime as dt
from typing import Union, List, Dict
import scipy.sparse as sp

from eaopack.basic_classes import Timegrid

class Results:
    def __init__(self, value: Union[float, None],
                       x:     Union[np.array, None],
                       duals: Union[dict, None]):
        """ collection of optimization results

        Args:
            value (float): value of optimized target function
            x (np.array): optimal values of variables
            duals (dict): dual values for  constraints. Dict with constraint types as keys (mostly interesting 'N' for nodes)
        """
        self.value = value
        self.x     = x
        self.duals = duals

class OptimProblem:

    def __init__(self, 
                 c: np.ndarray, 
                 l: np.ndarray, 
                 u: np.ndarray, 
                 A: Union[np.ndarray, None]                = None, 
                 b: Union[np.ndarray, None]                = None, 
                 cType:Union[str, None]                    = None,
                 mapping: Union[pd.DataFrame, None]        = None,
                 timegrid:Union[Timegrid, None]            = None,  # needed if periodic
                 periodic_period_length:Union[str, None]   = None,
                 periodic_duration:Union[str, None]        = None,
                 map_nodal_restr:Union[list, None]         = None 
                 ):
        """ Formulated optimization problem. LP problem.

        Args:
            c (np.array): cost vector
            l (np.array): lower bound (per time step)
            u (np.array): upper bound (per time step)
            A (np.array): restiction matrix. Optional. Defaults to None (no restrictions given)
            b (np.array): bound of restriction. Optional. Defaults to None (no restrictions given)
            cType (str - one letter per restriction): Logic to define type of restriction: U-pper, L-ower, S-equal or other 
                                                      here specific types may be defined:
                                                         sum of dispatch at node zero: N
                                                      Optional. Defaults to None (no restrictions given)
            mapping (pd.DataFrame): Mapping of variables to 'asset', 'node', 'type' ('d' - dispatch and 'i' internal variable) and 'time_step' 
            map_nodal_restr (list): Mapping to identify node and timepoint for duals in the solution of the LP

            --- if periodic
            timegrid (Timegrid)          : timegrid underneath optim problem (defaults to None)
            periodic_period_length (str) : pandas freq defining the period length (defaults to None)
            periodic_duration (str)      : pandas freq defining the duration of intervals in which periods are repeated (defaults to None - then duration is inf)
        """
        self.c     = c     # cost vector
        self.l     = l     # lower bound
        self.u     = u     # upper bound
        self.A     = A     # restriction matrix
        self.b     = b     # restriction result vector
        self.cType = cType # GLPK type of restriction (le, ge, ...)      
        self.mapping = mapping 
        self.map_nodal_restr = map_nodal_restr
        assert not np.isnan(c.sum()), 'nan value in optim problem. Check input data -- c'
        assert not np.isnan(l.sum()), 'nan value in optim problem. Check input data -- l'
        assert not np.isnan(u.sum()), 'nan value in optim problem. Check input data -- u'
        if not b is None: assert not np.isnan(b.sum()), 'nan value in optim problem. Check input data -- b'

        # make periodic
        if periodic_period_length is not None:
            assert timegrid is not None, 'for periodic optim problem need timegrid'
            self.__make_periodic__(freq_period = periodic_period_length , freq_duration = periodic_duration, timegrid = timegrid)

    def __make_periodic__(self, freq_period:str, freq_duration:str, timegrid:Timegrid):
        """ Make the optimization problem periodic main purpose is to save resources when optimizing 
            granular problems over a long time -- e.g. a year with hourly resolution -- where the
            finer resolution shows periodic behaviour -- e.g. typical load profiles over the day

            The routine is typically called during init of optim problem

        Args:
            freq_period (str): [description]
            freq_duration (str): [description]
        """

        # (1)  create mapping of timegrid to periodicity intervals #################################
        #      We create a numbering 0, 1, ... for each period
        #      and identify duration intervals. In case there's an overlap between periods and durations
        #      periods are leading. However - best choice are frequencies that don't create this problem

        # disp factor column needed to assign same dispatch to all related time steps
        if 'disp_factor' not in self.mapping.columns: self.mapping['disp_factor'] = 1.
        #self.mapping['disp_factor'].fillna(1., inplace = True) # ensure there's a one where not assigned yet  --> warning & deprecation pandas 3.0
        self.mapping['disp_factor'] = self.mapping['disp_factor'].fillna(1.) # ensure there's a one where not assigned yet
        tp = timegrid.timepoints
        T  = timegrid.T
        try: periods = pd.date_range(tp[0]-pd.Timedelta(1, freq_period),    tp[-1]+pd.Timedelta(1, freq_period), freq = freq_period, tz = timegrid.tz)
        except: periods = pd.date_range(tp[0]-pd.Timedelta(freq_period), tp[-1]+pd.Timedelta(freq_period),    freq = freq_period, tz = timegrid.tz)
        if freq_duration is None:
            durations = [tp[0], tp[-1]+(tp[-1]-tp[0])] # whole interval - generously extending end
        else:
            try: durations = pd.date_range(tp[0]-pd.Timedelta(1, freq_duration), tp[-1]+pd.Timedelta(1, freq_duration), freq = freq_duration, tz = timegrid.tz)
            except: durations = pd.date_range(tp[0]-pd.Timedelta(freq_duration), tp[-1]+pd.Timedelta(freq_duration), freq = freq_duration, tz = timegrid.tz)    
        # gave a bit space in case date ranges do not have the same start - deleting now superfluous (early) start
        if periods[1] <= tp[0]: periods = periods.drop(periods[0])
        if durations[1] <= tp[0]: durations = durations.drop(durations[0])
        # assertions - ensure that all periods are of same length
        d = periods[1:]-periods[0:-1]
        assert all(d==d[0]), 'Error. All periods must have same length. Not given for chosen frequency '+periods
        ### create df that assigns periods and duration intervals to all tp's
        df = pd.DataFrame(index = timegrid.I)
        df['dur']     = np.nan
        df['per']     = np.nan
        df['sub_per'] = np.nan
        i_dur = 0
        i_per = 0
        i_sub_per = 0
        # df['dur'].iloc[0] = 0
        # df['per'].iloc[0] = 0
        ind_dur = df.columns.get_indexer(['dur'])[0]
        ind_per = df.columns.get_indexer(['per'])[0]
        ind_sub_per = df.columns.get_indexer(['sub_per'])[0]
        df.iloc[0, ind_dur] = 0
        df.iloc[0, ind_per] = 0
        for i in range(0,T):
            if tp[i] >= durations[i_dur]: 
                i_dur +=1
                # df['dur'].iloc[i] = int(i_dur)
                df.iloc[i, ind_dur] = int(i_dur)
            if tp[i] >= periods[i_per]: 
                i_per +=1
                # df['per'].iloc[i] = int(i_per)
                df.iloc[i, ind_per] = int(i_per)
                i_sub_per = 0
            # df['sub_per'].iloc[i] = int(i_sub_per)
            df.iloc[i, ind_sub_per] = int(i_sub_per)
            i_sub_per += 1
                
        df.ffill(inplace = True)
        df['per']     = df['per'].astype(int)
        df['dur']     = df['dur'].astype(int)
        df['sub_per'] = df['sub_per'].astype(int)

        self.mapping = pd.merge(self.mapping, df, left_on = 'time_step', right_index = True, how = 'left')
        self.mapping['new_idx'] = self.mapping.index
        idx = np.asarray(self.mapping.index).copy() # get mapping index to change it later on
        # (2)  loop through each variable-group and group together all #################################
        #      elements that belong to the same period item
        all_out = [] # collect all vars to remove
        for myasset in self.mapping['asset'].unique():
            for mynode in self.mapping['node'].unique():
                for mytype in self.mapping['type'].unique():
                    for myvar in list(self.mapping['var_name'].unique()):
                        I = (self.mapping['asset'] == myasset)&(self.mapping['node'] == mynode)&(self.mapping['var_name'] == myvar)&(self.mapping['type'] == mytype)
                        # loop through durations
                        for dur in self.mapping.loc[I].dur.unique():
                            # loop through period steps
                            for sub_per in self.mapping.loc[I & (self.mapping['dur'] == dur)].sub_per.unique():
                                II = I & (self.mapping['dur'] == dur) & (self.mapping['sub_per'] == sub_per)
                                if II.sum() <= 1:
                                    pass ##  Nothing to do. There is only one variable
                                else:
                                    vars    = self.mapping.index[II] # variables to be joined
                                    leading = vars[0]       # this one to remain
                                    out     = vars[1:]
                                    all_out +=out.to_list()
                                    # shrink optimization problem
                                    ####  u, l, c
                                    # bounds should ideally be equal anyhow. here choose average
                                    self.l[leading] = self.l[vars].mean()
                                    self.u[leading] = self.u[vars].mean()
                                    # leading variable takes joint role - thus summing up costs
                                    self.c[leading] = self.c[vars].sum()
                                    #### if given, A (b and cType refer to restrictions)
                                    # need to add up A elements for vars to be deleted in A elements for leading var
                                    if self.A is not None:
                                        self.A = self.A.tolil()
                                        self.A[:,leading] += self.A[:,out].sum(axis = 1)
                                    # Adjust mapping. 
                                    assert all(self.mapping.loc[II, 'disp_factor'] == self.mapping.loc[II, 'disp_factor'].iloc[0]), 'periodicity cannot be imposed where disp factors are not identical'
                                    idx[out] = leading
                                    self.mapping.loc[out, 'new_idx'] = leading

        self.l = np.delete(self.l,all_out)
        self.u = np.delete(self.u,all_out)                                
        self.c = np.delete(self.c,all_out)
        if self.A is not None:
            my_idx = self.mapping.index.unique() # full index
            my_idx = np.delete(my_idx, all_out)
            self.A = self.A[:,my_idx]
        self.mapping.drop(columns = ['dur','per','sub_per'], inplace = True)
        self.mapping.set_index('new_idx', inplace = True)

    @property
    def is_MIP(self) -> Union[bool, None]:
        """ return flag, whether op is LP or MIP
            - None if not set up yet
            - True if MIP (there are binary/integer variables)
            - False if no binary/integer variables """
        if self.mapping is None: flag = None
        elif 'bool' in self.mapping.columns:
            flag = any(self.mapping['bool'])
        else: flag = False
        return flag


    def optimize(self, target        = 'value',
                       samples       = None,
                       interface:str = 'cvxpy', 
                       solver        = None,  
                       make_soft_problem=False,
                       solver_params=None) -> Union[Results, str]:
        """ optimize the optimization problem

        Args:
            target (str): Target function. Defaults to 'value' (maximize DCF). 
                                           Alternative: 'robust', maximizing the minimum DCF across given price samples
            samples (List): Samples to be used in specific optimization targets
                            - Robust optimization: list of costs arrays (maximizing minimal DCF)
            interface (str, optional): Chosen interface architecture. Defaults to 'cvxpy'. "ortools" also possible.
            solver (str, optional): Solver for interface. Defaults to None, in which case SCIP is utilized (currently best experience)
                                    choose "CVXPY" to let CVXPY decide (with interface "cvxpy")
                                    Note: CVXPY is used as interface to solvers. See details on solvers here:  https://www.cvxpy.org/tutorial/solvers/index.html
            make_soft_problem (bool, optional): If true, relax the boolean variables and allow float values instead. Defaults to False
        """
        assert np.all(self.l <= self.u), 'Error. Lower bounds must be smaller or equal to upper bounds'
        map = self.mapping.copy()  # abbreviation

        # relaxing MIP to LP simply by forcing all boolean variables to be continuous
        if make_soft_problem:
            map['bool'] = False
        isMIP = False
        if 'bool' in map:
            my_bools = map.loc[(~map.index.duplicated(keep='first')) & (map['bool'])].index.values.tolist()
            if len(my_bools) == 0:
                my_bools = False
            else:
                isMIP = True  ### !!! Need to change solver
        else:
            map['bool'] = False
            my_bools = False


        if interface == 'cvxpy':
            import cvxpy as CVX
            ########### default solver. For MIPs setting SCIP as default solver - currently best experience
            if solver is None:
                if isMIP: solver = "SCIP"  
            elif solver.upper() == "CVXPY":
                solver = None # let CVXPY decide

            # reformulate boolean var list - backwards compatibility due to change in CVXPY
            if isMIP:
                if (CVX.__version__>='1.6'): my_bools = (my_bools,)
                else: my_bools = [(bb,) for bb in my_bools]

            # Construct the problem
            x = CVX.Variable(self.c.shape, boolean=my_bools)
            ##### put together constraints
            constr_types = {}   # dict to remember constraint type and numbering to extract duals
            # lower and upper bound  constraints # 0 & 1
            constraints = [ x <= self.u, x>=self.l ]

            constr_types['bound_u'] = 0  # first CVX constraint 
            constr_types['bound_l'] = 1  # second CVX constraint ...
            counter_constr_type = 1 # keep track of number of constraint types to be able to identify

            if not self.A is None: 
                assert (len(self.b) == len(self.cType)) and (len(self.b) == self.A.shape[0]) and (len(self.u) == self.A.shape[1])
                #UPPER limit
                my_type = "U"
                # identify rows
                myRows = [mya==my_type for mya in self.cType] 
                self.A = self.A.tolil() # check - necessary? Leftover?
                if any(myRows):
                    counter_constr_type += 1
                    constr_types[my_type]  = counter_constr_type
                    myRows = np.asarray(myRows)
                    AU = self.A[myRows, :]
                    bU = np.asarray(self.b)
                    bU = bU[myRows]
                    constraints = constraints + [ AU @ x<=bU ] # matrix/vector multiplication in CVXPY notation
                #LOWER limit 
                my_type = "L"
                myRows = [mya==my_type for mya in self.cType] 
                if any(myRows):
                    counter_constr_type += 1
                    constr_types[my_type]  = counter_constr_type
                    myRows = np.asarray(myRows)
                    AL = self.A[myRows, :]
                    bL = np.asarray(self.b)
                    bL = bL[myRows]
                    constraints = constraints + [ AL @ x>=bL ]
                #EQUAL constraints 
                my_type = "S"
                myRows = [mya==my_type for mya in self.cType] 
                if any(myRows):
                    counter_constr_type += 1
                    constr_types[my_type]  = counter_constr_type
                    myRows = np.asarray(myRows)
                    AS = self.A[myRows, :]
                    bS = np.asarray(self.b)
                    bS = bS[myRows]
                    constraints = constraints + [ AS @ x==bS ]

                # Nodal constraints (special interpretation, but essentially type EQUAL)
                my_type = "N"
                myRows = [mya==my_type for mya in self.cType] 
                if any(myRows):
                    counter_constr_type += 1
                    constr_types[my_type]  = counter_constr_type
                    myRows = np.asarray(myRows)
                    AN = self.A[myRows, :]
                    bN = np.asarray(self.b)
                    bN = bN[myRows]
                    constraints = constraints + [ AN @ x==bN ]

            # Target function - alternatives possible
            if target.lower() == 'value':
                objective = -self.c.T @ x         # @ is the matrix/vector multiplication in CVXPY notation
            elif target.lower() == 'robust':
                assert (not samples is None) # need samples
                if (isinstance(samples[0], (float, int))):
                    raise ValueError('For robust optimization, samples must be list of arrays ')
                # (1) new variable, representing the minimum DCF
                DCF_min = CVX.Variable(1)
                # (2) each price sample represents a new restriction DCF_sample >= DCF_min
                for myc in samples:
                    constraints = constraints + [-myc.T @ x >= DCF_min ] # sign: maximize negative costs
                objective = DCF_min
            else:
                raise NotImplementedError('Target function -- '+target+' -- not implemented')
            prob = CVX.Problem(CVX.Maximize(objective), constraints)
          
            if solver is None:
                prob.solve()
            else:
                inst_solv = CVX.installed_solvers()
                if not solver in inst_solv:
                    raise ValueError('Solver '+solver+' not installed. Installed solvers are: '+str(inst_solv))
                prob.solve(solver = getattr(CVX, solver))                

            if prob.status == 'optimal':
                # print("Status: " +prob.status)
                # print('Portfolio Value: ' +  '% 6.0f' %prob.value)

                if not isMIP:
                    # collect duals in dictionary according to cTypes
                    myduals = {}
                    for myt in constr_types:
                        myduals[myt] = constraints[constr_types[myt]].dual_value
                else:
                    myduals = None
                results = Results(value       = prob.value,
                                  x           = x.value,
                                  duals = myduals)
                if target.lower() == 'robust':
                    # in case of robust target, the optimized value is the minimum
                    results.value = -sum(x.value * self.c)
            elif prob.status == 'optimal_inaccurate':
                print('Optimum found, but inaccurate: ' + prob.status)               
                results = 'inaccurate'
            else:
                print('Optimization not successful: ' + prob.status)       
                results = 'not successful'
        elif interface == "ortools":
            from ortools.linear_solver import pywraplp

            if solver is None:
                solver = "SCIP"
            solver_ = pywraplp.Solver.CreateSolver(solver)
            if not solver_:
                raise Exception("Something went wrong when creating the solver.")

            # Create variables:
            infinity = solver_.infinity()
            x = {}
            for j in range(len(self.l)):
                if isMIP and any(map.loc[[j]]["bool"]==True):
                    if self.u[j] != 1 or self.l[j] != 0:
                        x[j] = solver_.IntVar(self.l[j], self.u[j], "x[%i]" % j)
                    else:
                        x[j] = solver_.BoolVar("x[%i]" % j)
                else:
                    x[j] = solver_.NumVar(self.l[j], self.u[j], "x[%i]" % j)

            if not self.A is None:
                assert (len(self.b) == len(self.cType)) and (len(self.b) == self.A.shape[0]) and (len(self.u) == self.A.shape[1])
                self.A = self.A.tolil()

                for i in range(len(self.A.rows)):
                    if self.cType[i] == "U":
                        constraint = solver_.RowConstraint(-infinity, self.b[i], "")
                    elif self.cType[i] == "L":
                        constraint = solver_.RowConstraint(self.b[i], infinity, "")
                    elif self.cType[i] == "S" or self.cType[i] == "N":
                        constraint = solver_.RowConstraint(self.b[i], self.b[i], "")
                    else:
                        raise ValueError("Unknown ctype " + self.cType[i] +  "at position " + str(i))
                    for j in range(len(self.A.rows[i])):
                        constraint.SetCoefficient(x[self.A.rows[i][j]], self.A.data[i][j])

            # Set objective
            objective = solver_.Objective()
            for j in range(len(self.c)):
                objective.SetCoefficient(x[j], -self.c[j])
            objective.SetMaximization()

            print(f"Solving with {solver_.SolverVersion()}")

            # solver_.set_solver_specific_parameters({})
            if solver_params is not None:
                solver_.SetSolverSpecificParametersAsString(solver_params)
            status = solver_.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                print("Optimal solution found.")
                x_value = np.array([x[j].solution_value() for j in range(len(self.l))])
                results = Results(value=solver_.Objective().Value(),
                                  x=x_value,
                                  duals=None)
            elif status == pywraplp.Solver.FEASIBLE:
                print("A feasible solution was found, but we don't know if it's optimal.")
                x_value = np.array([x[j].solution_value() for j in range(len(self.l))])
                results = Results(value=solver_.Objective().Value(),
                                  x=x_value,
                                  duals=None)
            elif status == pywraplp.Solver.INFEASIBLE:
                print("The problem is infeasible.")
                results = 'infeasible'
            else:
                print('Optimization not successful: ' + str(status))
                results = 'not successful'
        else:
            raise NotImplementedError('Interface - '+str(interface)+ ' -not implemented')

        return results

class SplitOptimProblem(OptimProblem):
    def __init__(self, ops, mapping):
        """ Collection of consecutive OptimProblems

            Args:
                ops: List of OptimProblems
                mapping (pd.DataFrame): Mapping of all result variables to 'asset', 'node', 'type' ('d' - dispatch and 'i' internal variable) and 'time_step'
        """
        self.ops = ops
        self.mapping = mapping
        self.c = np.hstack([op.c for op in ops])
        if not ops[0].map_nodal_restr is None:
            self.map_nodal_restr = []
            for op in ops:
                self.map_nodal_restr += op.map_nodal_restr
        else:
             self.map_nodal_restr = None

    def optimize(self, *args, **kwargs) -> Results:
        """ Optimize all OptimProblems in self.ops and piece the results together in one Result
        """
        res = Results(0, np.array([]), None)
        for op in self.ops:
            res_tmp = op.optimize(*args, **kwargs)
            res.value += res_tmp.value
            res.x = np.hstack((res.x, res_tmp.x))
            if res_tmp.duals:
                if res.duals:
                    for key in res_tmp.duals:
                        if res.duals[key] is not None:
                            if res_tmp.duals[key] is None:
                                res.duals[key] = None
                            else:
                                res.duals[key] = np.hstack((res.duals[key], res_tmp.duals[key]))
                else:
                    res.duals = res_tmp.duals
        return res

