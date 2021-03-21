import pingouin as pg
data = pg.read_dataset('chi2_independence')

import pandas as pd


import seaborn as sns

data = sns.load_dataset("tips")
print(data.head())

expected, observed, stats = pg.chi2_independence(data, x='sex', y='day')

print(stats)