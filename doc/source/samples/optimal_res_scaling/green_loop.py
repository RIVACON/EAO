import numpy as np
import pandas as pd
import datetime as dt

# in case eao is not installed
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '../..'))
sys.path.append(join(mypath, '../../..'))
sys.path.append(join(mypath, '../../../..'))

import eaopack as eao
from eaopack.assets import  Contract, Storage, SimpleContract, ScaledAsset
from eaopack.basic_classes import Node, Timegrid, Unit
from eaopack.portfolio import Portfolio
from eaopack.serialization import to_json
from eaopack.io import extract_output, output_to_file

###############################################   setting
Start = dt.date(2020, 1, 1)
End   = dt.date(2021, 1, 1)
vec_min_fraction_green_ppa    = [.5, .55, .6, .65, .7,.75, .8,.85, .9, .95]
freq = 'h'
volume_sold    = 1e3 # volume of sales
# define file names for this sample
## input data
file_profiles      = join(mypath, 'load_profiles_DK.csv')
file_spot_prices   = join(mypath, 'elspotprices.csv')

## output files for results
file_normed_assets       = join(mypath, 'normed_assets.json')
file_scaling_assets      = join(mypath, 'scaling_assets.json')
file_vol_matching_assets = join(mypath, 'vol_matching_assets.json')
file_timegrid            = join(mypath, 'timegrid.json')
file_prices              = join(mypath, 'prices.json')

file_results              = join(mypath, 'loop_results')

###############################################   create helpers 
print('create timegrid and nodes')
timegrid = Timegrid(start = Start, end = End, freq = freq, main_time_unit= 'h')
to_json(timegrid, file_timegrid)
node_green = Node('green sources', commodity= 'power', unit = Unit('MWh', 'MW',  1.))
node_grey  = Node('grey sources', commodity= 'power', unit = Unit('MWh', 'MW',  1.))
node_load  = Node('client', commodity= 'power', unit = Unit('MWh', 'MW',  1.))

###############################################   import data
print('import load profiles and prices')
df_profiles = pd.read_csv(file_profiles)
#df_profiles.index = pd.to_datetime(df_profiles['HourUTC'], format='%Y-%m-%dT%H')
df_profiles.index = pd.to_datetime(df_profiles['HourUTC'], format='%Y-%m-%dT%H:00:00+00:00')
df_profiles.sort_index(inplace = True)
# filter for only DK1 price area and time grid
df_profiles = df_profiles.loc[df_profiles['PriceArea'] == 'DK1', ['OnshoreWindPower', 'OffshoreWindPower', 'SolarPower', 'TotalLoad']]
df_profiles = df_profiles.loc[Start:End]
# omit last
df_profiles = df_profiles.iloc[:-1]
# import prices
df_prices = pd.read_csv(file_spot_prices)
#df_prices.index = pd.to_datetime(df_prices['HourUTC'], format='%Y-%m-%dT%H')
df_prices.index = pd.to_datetime(df_prices['HourUTC'], format='%Y-%m-%dT%H:00:00+00:00')
df_prices.sort_index(inplace = True)
df_prices = df_prices.loc[df_prices['PriceArea']=='DK1', ['SpotPriceEUR']]
df_prices = df_prices.loc[Start:End]

# resample to time grid
df_prices = df_prices.resample(freq).mean()
df_profiles = df_profiles.resample(freq).mean()
df_prices.ffill(inplace = True)
df_profiles.ffill(inplace = True)
# normalize to maximum 1 --> normalized to 1 MW net peak capacity
df_profiles = df_profiles/df_profiles.max()
# normalize load to 100 GWh
df_profiles['TotalLoad'] = df_profiles['TotalLoad']*volume_sold/df_profiles['TotalLoad'].sum()

# scaling of RES to meet total volume
scale_res = {}
scale_res['pv']       = volume_sold/df_profiles['SolarPower'].sum()/2.
scale_res['onshore']  = volume_sold/df_profiles['OnshoreWindPower'].sum()/2.
#scale_res['offshore'] = volume_sold/df_profiles['OffshoreWindPower'].sum()

# cast prices into timegrid
prices =  {'start': df_prices.index.values, 'values': df_prices.SpotPriceEUR.values}
prices = {'spot': timegrid.values_to_grid(prices)}
to_json(prices,file_prices)

results_vals = []
for min_fraction_green_ppa in vec_min_fraction_green_ppa:
        max_fraction_green_sales  = 1. - min_fraction_green_ppa + .05

        ############################################## normed assets
        print('assets')
        print(' ... normalized wind and pv assets')
        pv_profile   = {'start': df_profiles.index.values, 'values': df_profiles.SolarPower.values}
        pv_normed    = Contract(name = 'pv', min_cap= pv_profile, max_cap= pv_profile, nodes = node_green)
        onshore_profile = {'start': df_profiles.index.values, 'values': df_profiles.OnshoreWindPower.values}
        onshore_normed  = Contract(name = 'onshore', min_cap= onshore_profile, max_cap= onshore_profile, nodes = node_green)
        offshore_profile = {'start': df_profiles.index.values, 'values': df_profiles.OffshoreWindPower.values}
        offshore_normed  = Contract(name = 'offshore', min_cap= offshore_profile, max_cap= offshore_profile, nodes = node_green)

        print(' ... a load profile to be delivered (here the DK1 total load scaled to a max of 10 MW')
        load_profile   = {'start': df_profiles.index.values, 'values': -df_profiles.TotalLoad.values}
        load           = Contract(name = 'load', min_cap= load_profile, max_cap= load_profile, nodes = node_load)

        max_load       = -load_profile['values'].min()

        print(' ... spot market to fill the gaps (with grey power)')
        spot_market = SimpleContract(name = 'spot', price = 'spot', min_cap = 0, max_cap = +1.1*max_load, nodes= node_grey)
        max_green_sales = {'start':Start, 'end':End, 'values' : -volume_sold * max_fraction_green_sales }
        print(' ... green sales. Potentially limited to limit capacity overbuild')
        green_sales = Contract(name = 'green spot', price = 'spot', min_cap = -20.*max_load, max_cap = 0, 
                        nodes= node_green, min_take=max_green_sales)

        print(' ... a battery storage with 90% cycle efficiency and costs of 1 EUR/MWh ')
        storage_normed = Storage(name = 'battery', cap_in= 1, cap_out=1, size = 4, \
                        eff_in=0.9, nodes = node_green, cost_in=1., cost_out=0, block_size = '4d')


        print(' ... links from load to  green and grey sources to implement minimimum green share')
        # define minimum sourcing from green sources 
        min_green = {'start':Start, 'end':End, 'values' : volume_sold * min_fraction_green_ppa }
        link_green = eao.assets.ExtendedTransport(name = 'link_green', min_take = min_green,  
                                                min_cap = 0, max_cap = 1.1*max_load, nodes = [node_green, node_load])
        link_grey  = eao.assets.Transport(name = 'link_grey',   min_cap = 0, max_cap = 1.1*max_load, nodes = [node_grey, node_load])

        print(' ... package downstream contract into PPA')
        downstream_portfolio = eao.portfolio.Portfolio([load, link_green, link_grey])
        downstream_contract  = eao.portfolio.StructuredAsset(name = 'downstream', nodes = [node_green, node_grey], portfolio = downstream_portfolio)


        print('write normed assets to file')
        portf = Portfolio([downstream_contract,
                        spot_market,
                        pv_normed,
                        onshore_normed,
                        offshore_normed,
                        green_sales])

        print('Scaling assets -- allowing to put together an optimally sized portfolio of all technologies')
        ## LCOEs for RES correspond to fix costs
        ## source IEA, at 7% interest, utility scale
        ##  in USD ... 1,2 USD/EUR
                ## PV 43,56        at cap factor 18%
                ## onshore 29,18   at cap factor 40%
                ## offshore 45,09  at cap factor 52%
                ## battery approx 1,2 M€/MW installed

        cost_factor = 1 ## to play around. should be 1
        max_scale = max_load*10
        fix_costs = cost_factor*43.56/1.2*0.18 # get fix costs per main time unit (='h') from LCOE
        pv        = ScaledAsset(name = 'pv', base_asset = pv_normed, max_scale= max_scale, fix_costs= fix_costs)
        fix_costs = cost_factor*29.18/1.2*0.4  # get fix costs per main time unit (='h') from LCOE
        onshore   = ScaledAsset(name = 'onshore', base_asset = onshore_normed, max_scale = max_scale, fix_costs= fix_costs)
        #fix_costs = cost_factor*45.09/1.2*0.52 # get fix costs per main time unit (='h') from LCOE
        #offshore  = ScaledAsset(name = 'offshore', base_asset = offshore_normed, max_scale = max_scale, fix_costs= fix_costs)
        fix_costs = 90000/8760 # rough estimate - fix costs of a battery
        storage   = ScaledAsset(name = 'battery', base_asset = storage_normed, max_scale = max_scale, fix_costs= fix_costs)


        ############################################## fully scaling all assets
        print('optimize')
        print(str(min_fraction_green_ppa))
        portf = Portfolio([downstream_contract,
                        spot_market,
                        pv,
                        onshore,
                        storage,
                        green_sales])

        op  = portf.setup_optim_problem(prices = prices, timegrid=timegrid)
        res = op.optimize()
        if not isinstance(res, str):
                out = extract_output(portf, op, res)
                output_to_file(out, file_name = file_results + str(int(100*(min_fraction_green_ppa)))+'.xlsx')
                results_vals.append(res.value)
        else:
                results_vals.append('infeasible')
df = pd.DataFrame(index = vec_min_fraction_green_ppa)
df['values'] = results_vals
writer = pd.ExcelWriter(file_results+'_values.xlsx')
df.to_excel(writer)
writer.close()
