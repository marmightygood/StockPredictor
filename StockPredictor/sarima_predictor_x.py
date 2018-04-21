import pandas as pd
from pandas.io.parsers import read_csv
import numpy as np
import sys
import os

from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

from pandas import DataFrame
from statsmodels.tsa.statespace.sarimax import SARIMAX
 
def parser(x):
	return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

symbol = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
model_dir = sys.argv[4]

logf = open(model_dir + "sarima_predictor_" + str(os.getpid()) + ".log", "a")

print ("Loading symbol {0} from {1}", symbol, input_dir)

#Load data
data = pd.read_csv(input_dir + symbol + '_prices.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser, usecols=[1,2])
resampled = data.resample('M').mean().bfill()

#Read the best parameters (calculated earlier)
#parameters = pd.read_csv(model_dir + symbol + '_sarimax_best_parameters.csv')
ar=1
i=0
ma=1
s=6

#Fit model
model = SARIMAX(resampled, order=(ar,i,ma), seasonal_order=(2,1,1,s), enforce_stationarity=False, enforce_invertibility = False )
model_fit = model.fit()

#Create output dataset
output = model_fit.forecast(6)

output = pd.DataFrame(output)
output.columns = ['PredictedPrice']
                
output.index.name = 'LastTradeDateKey'
output.to_csv(output_dir + symbol + '_sarimax_predictions.csv')
    
# plot
pyplot.plot(data, color='blue')
pyplot.plot(output, color='red')
pyplot.show()

