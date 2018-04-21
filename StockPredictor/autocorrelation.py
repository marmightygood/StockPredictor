import pandas as pd
from pandas.io.parsers import read_csv
import numpy as np
import sys

from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
#from matplotlib import pyplot

from sklearn.metrics import mean_squared_error
 
def parser(x):
	return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

symbol = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]

print ("Loading symbol {0} from {1}", symbol, input_dir)

#Load data
data = pd.read_csv(input_dir + symbol + '_prices.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser, usecols=[1,2])

pyplot.plot(data)
pyplot.show()

pyplot.figure()
autocorrelation_plot(data)
pyplot.show()