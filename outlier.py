import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
from adtk.detector import LevelShiftAD
from adtk.visualization import plot

data = pd.read_excel('/home/siddique/Downloads/CPU_ransomware.xlsx', parse_dates=True, squeeze=True)
#s = validate_series(data)
data = pd.DataFrame(data)
#print(data)
raw_data = data.values
raw_data = raw_data[raw_data<=100]
data = raw_data[raw_data>=0]
data = pd.DataFrame(data)
data.to_csv('filtered_data.csv', header=0, index=False)
data = pd.read_csv('filtered_data.csv', header=0)
# data.reset_index(drop=True, inplace=True)
print(data)

data_time_index = pd.date_range("2022-01-01", periods=len(data.values), freq="H")
data = pd.Series(np.squeeze(data.values), index=data_time_index)#
#data = pd.DataFrame(list(zip(pd.DatetimeIndex(data=range(0,len(data.values))), data.values)))
#data.index.name = 'index'
# print(data[data.isnull()])
data = data.dropna()
#print(data.describe())
level_shift_ad = LevelShiftAD(c=6.0, side='both', window=32)
anomalies = level_shift_ad.fit_detect(data)
print(anomalies)

figure = plot(data, anomaly=anomalies, anomaly_color='red')
figure[0].get_figure().savefig('output_32.png')



