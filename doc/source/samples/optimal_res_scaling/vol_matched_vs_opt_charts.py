import numpy as np
import pandas as pd
import datetime as dt

# in case eao is not installed
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '../../../..'))
import eaopack as eao

import matplotlib.pyplot as plt

file_res = join(mypath, 'results_report/summary_vol_matched_optimized.xlsx')
file_chart= join(mypath, 'results_report/summary_vol_matched_optimized.pdf')


file_extract = join(mypath, 'results_report/extract_vol_matched.xlsx')
file_chart_extract = join(mypath, 'results_report/extract_vol_matched.pdf')
df = pd.read_excel(file_res, engine='openpyxl')
# cols: volume_matched     optimized



plt.rcdefaults()
fig, ax = plt.subplots(1,2, figsize=(8,4))
width = 0.35  # the width of the bars

################## capacity
ddf = df[df['type'] == 'cap']
x  = np.arange(len(ddf))
#yr = range(0,)
#yl = ['']*len(yr)
y1 = ddf['volume_matched']*1000
y2 = ddf['optimized']*1000

bar1 = ax[0].bar(x-width/2,y1,width, label = 'volume matched')
bar2 = ax[0].bar(x+width/2,y2,width, label = 'optimized')
#ax[0].set_xlabel('capacity')
ax[0].set_xticks(x)
ax[0].set_xticklabels(ddf['label'])
#ax.set_yticks(yr)
#ax.set_yticklabels(yl)
ax[0].set_ylabel('capacity [kW]')
ax[0].legend(loc = 'lower left')

################## costs
ddf = df[df['type'] == 'costs/kWh']
x  = np.arange(len(ddf))
#yr = range(0,)
#yl = ['']*len(yr)
y1 = ddf['volume_matched']
y2 = ddf['optimized']

bar1 = ax[1].bar(x-width/2,y1,width, label = 'volume matched')
bar2 = ax[1].bar(x+width/2,y2,width, label = 'optimized')
#ax[0].set_xlabel('capacity')
ax[1].set_xticks(x)
ax[1].set_xticklabels(ddf['label'])
ax[1].set_ylabel("costs [ct/kWh]")
#ax[0].legend(loc = 'lower left')

fig.tight_layout()
plt.savefig(file_chart)
#plt.show()


################### extract
df = pd.read_excel(file_extract, engine='openpyxl')
df.set_index('date', inplace = True)
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(8,4))

x  = df.index.hour

ax.plot(x,df['load'], label = 'load')
ax.plot(x,df['pv'], label = 'PV')
ax.plot(x,df['wind'], label = 'wind')
ax.plot(x,df['battery'], label = 'battery')
ax.plot(x,df['spot'], label = 'spot')

#ax[0].set_xlabel('capacity')
#ax[0].set_xticks(x)
#ax[0].set_xticklabels(ddf['label'])
#ax.set_yticks(yr)
#ax.set_yticklabels(yl)
ax.set_ylabel('production [kW]')
ax.set_xlabel('hour (25/06/2020)')
ax.set_xlim([0,23])
ax.grid(b=True, which='major', color='#666666', linestyle=':')
ax.legend(loc = 'upper right')
plt.savefig(file_chart_extract)