import numpy as np
import pandas as pd
import datetime as dt
import abc
import scipy.sparse as sp
from copy import deepcopy
from typing import Union, List, Tuple, Dict, Sequence
from eaopack.assets import Node, Asset, Timegrid, convert_time_unit
from eaopack.optimization import OptimProblem, SplitOptimProblem
from eaopack.optimization import Results 

class Portfolio:
    """ The portfolio class allows for collecting several assets in a network of nodes 
        and optimizing them jointly. In terms of setting up the problem, the portfolio
        collects the assets and imposes the restriction of forcing the flows of a commodity
        in each node to be zero in each time step  """

    def __init__(self, assets: List[Asset]):
        """ The portfolio class allows for collecting several assets in a network of nodes 
            and optimizing them jointly. In terms of setting up the problem, the portfolio
            collects the assets and imposes the restriction of forcing the flows of a commodity
            in each node to be zero in each time step.

            Args:
                assets (List[Asset]): Collection of the assets. The assets are assigned to nodes, which,
                                        together with 'Transport' assets define the network of the portfolio. """
        
        # collect some basic information from assets
        self.asset_names = []
        self.nodes       = {}
        for ia, a in enumerate(assets):
            assert isinstance(a, Asset), 'Portfolio mus consist of assets. Please check asset no. '+str(ia)
            self.asset_names.append(a.name)
            for n in a.nodes:
                if n.name not in self.nodes:
                    self.nodes[n.name] = n
        # some consistency checks
        assert (len(self.asset_names) == len(set(self.asset_names))), 'Asset names in portfolio must be unique'
        self.assets = assets

    def set_timegrid(self, timegrid:Timegrid):
        """ Set the timegrid for the portfolio. The timegrid will be used for all asset in the portfolio.
        Args:
            timegrid (Timegrid): The timegrid to be set
        """
        self.timegrid = timegrid

    def setup_optim_problem(self, prices: dict = None, 
                            timegrid:Timegrid = None, 
                            costs_only:bool = False, 
                            skip_nodes:list=[],
                            fix_time_window: Dict = None) -> OptimProblem:
        """ Set up optimization problem for portfolio

        Args:
            prices (dict): Dictionary of price arrays needed by assets in portfolio. Defaults to None
            timegrid (Timegrid, optional): Discretization grid for portfolio and all assets within. 
                                           Defaults to None, in which case it must have been set previously
            costs_only (bool): Only create costs vector (speed up e.g. for sampling prices). Defaults to False  
            skip_nodes (List): Nodes to be skipped in nodal restrictions (defaults to [])

            fix_time_window (Dict): Fix results for given indices on time grid to given values. Defaults to None
                           fix_time_window['I']: Indices on timegrid or alternatively date (all dates before date taken)
                           fix_time_window['x']: Results.x that results are to be fixed to in time window(full array, all times)

        Returns:
            OptimProblem: Optimization problem to be used by optimizer
        """
        ################################################## checks and preparations
        # set timegrid if given as optional argument
        if not timegrid is None:
            self.set_timegrid(timegrid)
        # check: timegrid set?        
        if not hasattr(self, 'timegrid'): 
            raise ValueError('Set timegrid of portfolio before creating optim problem.')        
        ################################################## set up optim problems for assets
        ## bounds
        l = np.array([])
        u = np.array([])
        c = np.array([])
        opt_probs = {} # dictionary to collect optim. problem for each asset
        mapping   = pd.DataFrame() # variable mappings collected from assets
        for a in self.assets:
            opt_probs[a.name] = a.setup_optim_problem(prices = prices, timegrid = self.timegrid, costs_only = costs_only)
            if not costs_only:
                mapping = pd.concat([mapping, opt_probs[a.name].mapping])
                # add together bounds and costs
                l = np.concatenate((l,opt_probs[a.name].l ), axis=None)
                u = np.concatenate((u,opt_probs[a.name].u ), axis=None)            
                c = np.concatenate((c,opt_probs[a.name].c ), axis=None)            
            else:
                c = np.concatenate((c,opt_probs[a.name]), axis=None)            
        if costs_only:
            return c        
        n_vars  = len(l)          # total number of variables
        n_nodes = len(self.nodes) # number of nodes
        T = self.timegrid.T       # number of time steps
        # new index refers to portfolio
        mapping.index.name = None
        mapping.reset_index(inplace = True) 
        mapping.rename(columns={'index':'index_assets'}, inplace=True) 
        #### mapping may come with several rows per variable 
        ## ensure index refers to variables: go through assets and their index, make unique
        mapping['keys'] = mapping['index_assets'].astype(str) +mapping['asset'].astype(str)
        idx = pd.DataFrame()
        idx['keys'] = mapping['keys'].unique()
        idx.reset_index(inplace = True)
        mapping = pd.merge(mapping, idx, left_on = 'keys', right_on = 'keys', how = 'left')
        mapping.drop(columns = ['keys'], inplace = True)
        mapping.set_index('index', inplace = True)
        ################################################## put together asset restrictions
        A = sp.lil_matrix((0, n_vars)) # sparse format to incrementally change
        b = np.zeros(0)
        cType = ''
        for a in self.assets:
            if not opt_probs[a.name].A  is None:
                n,m = opt_probs[a.name].A.shape
                ind = mapping.index[mapping['asset']==a.name].unique()
                myA = sp.lil_matrix((n, n_vars))
                myA[:,ind] = opt_probs[a.name].A
                opt_probs[a.name].A = None # free storage
                A = sp.vstack((A, myA))
                b = np.hstack((b, opt_probs[a.name].b))
                cType = cType + opt_probs[a.name].cType
        ################################################## create nodal restriction (flows add to zero)
        # record mapping for nodal restrictions to be able to assign e.g. duals to nodes and time steps
        # some assets work with a mapping column "disp_factor" that allows to account for a disp variable
        # only up to a factor (example transport; higher efficiency in setting up the problem)
        if 'disp_factor' not in mapping.columns:
            mapping['disp_factor'] = 1.
        # mapping['disp_factor'].fillna(1., inplace = True)   ### deprecated in pandas
        mapping['disp_factor'] = mapping['disp_factor'].fillna(1.)
        # mapping['nodal_restr'] = None

        def create_nodal_restr(nodes, map_nodes, map_types, map_idx, map_dispf, map_times, timegrid_I, skip_nodes, n_vars):
            """ Specific function creating nodal restrictions """
            map_nodal_restr = np.zeros(map_idx.shape[0])
            n_nodal_restr = 0
            cols = np.zeros(0)
            rows = np.zeros(0)
            vals = np.zeros(0)
            nodal_restr_map_expl = []
            for n in nodes:
                if (skip_nodes is None) or (not n in skip_nodes):
                    Inode = (map_types=='d') & (map_nodes==n)
                    for t in timegrid_I:
                        # identify variables belonging to this node n and time step t
                        I = (map_times[Inode] == t)
                        if I.sum()>0: # only then restriction needed
                            # myA = sp.lil_matrix((1, n_vars)) # one row only
                            # myA[0, map_idx[Inode][I]] = map_dispf[Inode][I]
                            newcols = map_idx[Inode][I]
                            cols = np.append(cols,newcols)
                            rows = np.append(rows, n_nodal_restr*np.ones(len(newcols)))
                            vals = np.append(vals, map_dispf[Inode][I])
                            Itemp = Inode.copy()
                            Itemp[Itemp] = I
                            map_nodal_restr[Itemp] = n_nodal_restr
                            n_nodal_restr +=1
                            nodal_restr_map_expl.append((t, n))
                            # A = sp.vstack((A, myA))
            return cols, rows, vals, nodal_restr_map_expl, n_nodal_restr

        # # easily readable version -  loop
        # perf = time.perf_counter()
        # n_nodal_restr = 0
        # for n in self.nodes:
        #     if not n in skip_nodes:
        #         for t in self.timegrid.I:
        #             # identify variables belonging to this node n and time step t
        #             I = (mapping['type']=='d') & \
        #                 (mapping['node']==n)   & \
        #                 (mapping['time_step'] == t).values
        #             if any(I): # only then restriction needed
        #                 myA = sp.lil_matrix((1, n_vars)) # one row only
        #                 myA[0, mapping.index[I]] = mapping.loc[I, 'disp_factor'].values ## extended with disp_factor logic
        #                 mapping.loc[I, 'nodal_restr'] = n_nodal_restr
        #                 n_nodal_restr +=1
        #                 A = sp.vstack((A, myA))
        # print('loop 1  duration '+'{:0.1f}'.format(time.perf_counter()-perf)+'s')
        ### start cryptic but much faster version, all in numpy
        map_nodes = mapping['node'].values
        map_types = mapping['type'].values
        map_idx = mapping.index.values
        map_dispf = mapping['disp_factor'].values
        map_times = mapping['time_step'].values
        if len(skip_nodes) == 0:
            my_skip_nodes = None
        else:
            my_skip_nodes = skip_nodes
        cols, rows, vals, map_nodal_restr, n_nodal_restr = create_nodal_restr(list(self.nodes.keys()), 
                                                                                map_nodes, 
                                                                                map_types, map_idx, map_dispf, 
                                                                                map_times, self.timegrid.I,my_skip_nodes, 
                                                                                n_vars)
        A = sp.vstack((A, sp.csr_matrix((vals, (rows.astype(np.int64), cols.astype(np.int64))), shape = (n_nodal_restr, n_vars))))
        # mapping['nodal_restr'] = map_nodal_restr.astype(np.int64)
        ### end cryptic version

        b = np.hstack((b,np.zeros(n_nodal_restr))) # must add to zero
        cType = cType + ('N')*n_nodal_restr

        # in case a certain time window is to be fixed, set l and u to given value
        # potentially expensive, as variables remain variable. however, assuming
        # this is fixed in optimization
        if not fix_time_window is None:
            assert 'I' in fix_time_window.keys(), 'fix_time_window must contain key "I" (time steps to fix)'
            assert 'x' in fix_time_window.keys(), 'fix_time_window must contain key "x" (values to fix)'
            if isinstance(fix_time_window['I'], (dt.date, dt.datetime)):
                fix_time_window['I'] = (timegrid.timepoints<= pd.Timestamp(fix_time_window['I']))
            assert (isinstance(fix_time_window['I'], (np.ndarray, list))), 'fix_time_window["I"] must be date or array'
            # in case of SLP, the problems may not be of same size (SLP is extended problem)
            # ---> then cut x to fix to size of the problem
            assert len(fix_time_window['x']) >= n_vars, 'fixing: values to fix appear to have the wrong size'
            if len(fix_time_window['x']) > n_vars:
                fix_time_window['x'] = fix_time_window['x'][0:n_vars]
            # get index of variables for those time points
            I = mapping['time_step'].isin(timegrid.I[fix_time_window['I']])
            l[I] = fix_time_window['x'][I]
            u[I] = fix_time_window['x'][I]
        return OptimProblem(c = c, l = l, u = u, A = A, b = b, cType = cType, mapping = mapping, map_nodal_restr = map_nodal_restr)

    def setup_split_optim_problem(self, prices: dict = None,
                                  timegrid: Timegrid = None,
                                  interval_size: str = 'd',
                                  skip_nodes: list = [],
                                  fix_time_window: Dict = None):
        """ Set up a split optimization problem for portfolio, i.e. split the timegrid into intervals of size
            interval_size and create a separate optimization problem for each interval

        Args:
            prices (dict): Dictionary of price arrays needed by assets in portfolio. Defaults to None
            timegrid (Timegrid, optional): Discretization grid for portfolio and all assets within.
                                           Defaults to None, in which case it must have been set previously
            interval_size (bool): Interval size according to pandas notation ('15min', 'h', 'd', ...). Defaults to 'd'
            skip_nodes (List): Nodes to be skipped in nodal restrictions (defaults to [])
            fix_time_window (Dict): Fix results for given indices on time grid to given values. Defaults to None
                           fix_time_window['I']: Indices on timegrid or alternatively date (all dates before date taken)
                           fix_time_window['x']: Results.x that results are to be fixed to in time window(full array, all times)

            Returns:
                SplitOptimProblem: A Split Optimization problem
        """
        interval_timepoints = pd.date_range(start=timegrid.start, end=timegrid.end, freq=interval_size,
                                            tz=timegrid.tz)
        interval_timepoints = interval_timepoints.append(pd.to_datetime([timegrid.end]))
        if interval_timepoints[0] != pd.Timestamp(timegrid.start):
            interval_timepoints = interval_timepoints.insert(0, timegrid.start)
        prices = timegrid.prices_to_grid(prices)
        ops = []
        mappings = []
        t = 0
        len_res = 0
        for i in range(len(interval_timepoints) - 1):
            start_tmp = interval_timepoints[i]
            end_tmp = interval_timepoints[i + 1]
            timegrid_tmp = Timegrid(start_tmp, end_tmp, timegrid.freq, ref_timegrid=timegrid)
            if timegrid_tmp.T == 0: continue
            tmp_I = timegrid_tmp.I # the original time steps
            timegrid_tmp.I = np.array(range(0, timegrid_tmp.T))  
            # timegrid_tmp.Dt = np.cumsum(timegrid_tmp.dt)           ### use the Dt from the reference --> for discounting
            prices_tmp = timegrid_tmp.prices_to_grid(prices)
            op_tmp = self.setup_optim_problem(prices_tmp, timegrid_tmp, skip_nodes=skip_nodes,
                                              fix_time_window=fix_time_window)
            mapping_tmp = deepcopy(op_tmp.mapping)
            # mapping_tmp["time_step"] += t
            # write original time step IDs to mapping
            orig_I = [tmp_I[a] for a in mapping_tmp["time_step"]]
            mapping_tmp["time_step"] = orig_I.copy()
            if i > 0:
                # mapping_tmp["nodal_restr"] += mappings[-1]["nodal_restr"].max()
                for asset in self.assets:
                    shift = mappings[-1][mappings[-1]["asset"] == asset.name]["index_assets"].max() + 1
                    mapping_tmp.loc[mapping_tmp["asset"] == asset.name, "index_assets"] += shift
            mapping_tmp.index += len_res
            mappings.append(mapping_tmp)
            # adjust numbering for time steps in map_nodal_restr
            tmp_nod_rest = []
            for mynr in op_tmp.map_nodal_restr:
                tmp_nod_rest.append((tmp_I[mynr[0]], mynr[1]))
            op_tmp.map_nodal_restr = tmp_nod_rest
            ops.append(op_tmp)
            len_res += op_tmp.c.shape[0]
            t += timegrid_tmp.T
        mapping = pd.concat(mappings)
        op = SplitOptimProblem(ops, mapping)
        self.set_timegrid(timegrid)
        for a in self.assets:
            a.set_timegrid(timegrid)
        return op

    def create_cost_samples(self, price_samples: List, timegrid: Timegrid = None) -> List:
        """ create costs vectors for LP on basis of price samples
        Args:
            price_samples (list): List of dicts of price arrays
            timegrid (Timegrid, optional): Discretization grid for portfolio and all assets within. 
                                        Defaults to None, in which case it must have been set previously

        Returns:
            list of costs vectors for use in OptimProblem (e.g. for robust optimization)  
        """

        res = []
        for ps in price_samples:
            res.append(self.setup_optim_problem(ps, timegrid, costs_only = True))
        return res

    def get_asset(self, asset_name: str) -> Asset:
        """ Return the asset with name asset_name or None if no asset with this name exists in the portfolio.

            Args:
                asset_name(str): The name of the asset

            Returns:
                asset (Asset): The asset with name asset_name
        """
        if asset_name in self.asset_names:
            idx = self.asset_names.index(asset_name)
            return self.assets[idx]
        else:
            return None

    def get_node(self, node_name: str) -> Node:
        """ Return the node with name node_name or None if no nodes with this name exists in the portfolio.

            Args:
                node_name(str): The name of the node

            Returns:
                node (Node): The node with name node_name
        """
        if node_name in self.nodes:
            return self.nodes[node_name]
        else:
            return None

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



if __name__ == "__main__" :
    pass
