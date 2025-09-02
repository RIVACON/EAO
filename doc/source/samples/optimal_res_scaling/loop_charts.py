import numpy as np
import pandas as pd
import datetime as dt

# in case eao is not installed
from os.path import dirname, join
import sys
mypath = (dirname(__file__))
sys.path.append(join(mypath, '../../../..'))
sys.path.append(join(mypath, '../../'))
import eaopack as eao

import matplotlib.pyplot as plt



###############################################   setting
Start = dt.date(2020, 1, 1)
End   = dt.date(2021, 1, 1)
freq = 'h'
volume_sold    = 1e3 # volume of sales
# define file names for this sample
## input data

file_results              = join(mypath, 'loop_results')
file_chart              = join(mypath, 'chart_loop_results.pdf')

df = pd.read_excel(file_results+'_values.xlsx',engine='openpyxl')
df = df[:-1]
df['values'] *=-1
df['values'] /= df['values'].max()
df.rename(columns = {df.columns[0]:'green_share', 'values': 'cost'}, inplace = True)

df['green_labels'] = (df['green_share']*100).apply(lambda x: '{0:.0f}'.format(x)+'%')
df['empty'] = ''
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(6,4))

x  = range(0,len(df))
yr = np.linspace(0,df['cost'].max(),10)
yl = ['']*len(yr)
y = df['cost']

ax.bar(x,y)
ax.set_xlabel('Share of green power')
ax.set_xticks(x)
ax.set_xticklabels(df['green_labels'])
ax.set_yticks(yr)
ax.set_yticklabels(yl)
ax.set_ylabel('Cost (normalized)')

plt.savefig(file_chart)
