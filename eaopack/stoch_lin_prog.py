import numpy as np
import pandas as pd
import datetime as dt
from typing import Union, List, Dict
import scipy.sparse as sp
from  copy import deepcopy

from eaopack.assets import Timegrid
from eaopack.portfolio import Portfolio
from eaopack.optimization import OptimProblem

def make_slp(optim_problem:OptimProblem, portf:Portfolio, timegrid:Timegrid,  start_future:dt.datetime, samples: List[Dict]) -> OptimProblem:
    """ Create a two stage SLP (stochastic linear program) from a given OptimProblem

    Args:
        optim_problem (OptimProblem)   : start problem
        portf (Portfolio)              : portfolio that is the basis of the optim_problem (e.g. to translate price samples to effect on LP)
        timegrid      (TimeGrid)       : timegrid consistent with optimproblem
        start_future  (dt.datetime)    : divides timegrid into present with certain prices 
                                            and future with uncertain prices, represented by samples
        samples (List[Dict])           : price samples for future. 
                                         (!) the future part of the original portfolio is added as an additional sample

    Returns:
        OptimProblem: Two stage SLP formulated as OptimProblem
    """
    assert pd.Timestamp(start_future) < pd.Timestamp(timegrid.end), 'Start of future must be before end for SLP'
    # (1) identify present and future on timegrid
    # future
    timegrid.set_restricted_grid(start = start_future)
    future_tg = deepcopy(timegrid.restricted)
    # present
    timegrid.set_restricted_grid(end = start_future)
    present_tg = deepcopy(timegrid.restricted)

    #### abbreviations
    # time grid
    T  = timegrid.T
    Tf = future_tg.T
    Tp = present_tg.T
    #ind_f = future_tg.I[0]  # index of start_future in time grid
    # number of samples
    nS = len(samples) 
    # optim problem
    n,m = optim_problem.A.shape

    # The SLP two stage model is the following:
        # \begin{eqnarray}
        #    \mbox{min}\left[ \bc^{dT} \bx^d + \frac{1}{S} \sum_s \hat \bc^{dsT} \bx^{ds}  \right] \\
        #    \mbox{with}\; A^s \colvec{\bx^d}{\hat\bx^{ds}}  \le \colvec{\bb^d}{\hat\bb^{ds}} \;\;\forall s = 1\dots S
        # \end{eqnarray}


    # (2) map variables to present & future --- and extend future variables by number of samples
    # the mapping information enables us to map variables to present and future and extend the problem
    # for future values, the dispatch information becomes somewhat irrelevant, but will
    # have an effect on decisions for the present
    slp_col = 'slp_step_'+str(future_tg.I[0])
    optim_problem.mapping[slp_col] = np.nan 
    ### mapping may contain duplicate rows for variables. Drop those
    temp_df = optim_problem.mapping[~optim_problem.mapping.index.duplicated(keep='first')]
    If      = temp_df['time_step'].isin(future_tg.I)   # does variable belong to future?
    del temp_df
    # future part of original cost vector gets number -1 
    optim_problem.mapping.loc[If,slp_col] = -1 
    map_f   = optim_problem.mapping.loc[If,:].copy()
    n_f      = len(map_f.index.unique())  #### now allowing for duplicate rows ... before len(map_f) # number of future variables
    #n_p      = m-n_f       # number of present variables
    # concatenate for each sample
    for i in range(0,nS):
        map_f[slp_col] = i
        optim_problem.mapping = pd.concat((optim_problem.mapping, map_f))
    optim_problem.mapping.reset_index(drop = True, inplace=True)
            ### aendern, falls auch nur samples für Zukunft gewuenscht ... so gehts nicht
            # # (3) translate price samples for future to cost samples (c vectors in LP)
            #     # The portfolio can only build the full LP (present & future). Thus we need to
            #     # append future samples with the (irrelevant) present prices, build the c's
            #     # and then ignore the present part. 
            #     # In case the length of the samples is already the full timegrid, step is ignored
            # for i, mys in enumerate(samples):
            #     for myk in mys:
            #         if len(mys[myk]) == T: 
            #             pass
            #         elif len(mys[myk]) == Tf:
            #             samples[i][myk] = np.hstack((optim_problem.c[:ind_f], mys[myk]))
            #         else:
            #             raise ValueError('All price samples for future must have length of full OR future part of timegrid') 
    c_samples = portf.create_cost_samples(price_samples = samples, timegrid = timegrid)

    # (4) extend LP (A, b, l, u, c, cType)
    #### Reminder: The original cost vector is interpreted as another sample. 

    ## extend vectors with nS times the future (the easy part)
    optim_problem.l = np.hstack((optim_problem.l, np.tile(optim_problem.l[If], nS)))
    optim_problem.u = np.hstack((optim_problem.u, np.tile(optim_problem.u[If], nS)))
    ## Attention with the cost vector. In order to obtain the MEAN across samples, divide by (nS+1) [new samples plus original]
    optim_problem.c[If] = optim_problem.c[If]/(nS+1) # orig. future sample
    for myc in c_samples: # add each future cost sample (and scale down to get mean)
        optim_problem.c = np.hstack((optim_problem.c, myc[If]/(nS+1)))

    ## different logic - restrictions simply multiply in number
    optim_problem.b = np.tile(optim_problem.b, nS+1)
    optim_problem.cType = optim_problem.cType*(nS+1)

    ## extending A & b (the trickier part)
    optim_problem.A = sp.lil_matrix(optim_problem.A) # convert to subscriptable format 
    # Note: Check and ideally avoid any such conversion (agree on one format)
    # futures only matric
    Af               = optim_problem.A[:, If]
    # "present only" matrix -- set future elements to zero to decouply
    Ap               = optim_problem.A.copy()
    Ap[:,If]         = 0.
    # start extending the matrix
    optim_problem.A  = sp.hstack((optim_problem.A, sp.lil_matrix((n, nS * n_f))))
    #### add  rows, that encode the same restriction as the orig. A for the orig. set - only with new set of future vars
    for i in range(0,nS):
        myA              = sp.hstack((Ap, sp.lil_matrix((n, (i)*n_f)), Af, sp.lil_matrix((n, (nS-i-1) * n_f))))
        optim_problem.A  = sp.vstack((optim_problem.A, myA))
    
    return optim_problem

