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

print ("Loading symbol {0} from {1}".format(symbol, input_dir))

#Load data
data = pd.read_csv(input_dir + symbol + '_prices.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser, usecols=[1,2])
data = data.resample('M').mean().bfill()

#Read the best parameters (calculated earlier)
try:
    parameters = pd.read_csv(model_dir + symbol + '_sarimax_best_parameters.csv')
    ar=np.asscalar(parameters['ar'].iloc[0])
    i=np.asscalar(parameters['i'].iloc[0])
    ma=np.asscalar(parameters['ma'].iloc[0])
    s=np.asscalar(parameters['s'].iloc[0])

    #Fit model
    model = SARIMAX(data, order=(ar,i,ma), seasonal_order=(2,1,1,s), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()

    #Create output dataset
    pred = model_fit.get_forecast(12, dynamic=False)


    #forecast = output.forecasts
    confidence = model_fit.conf_int(alpha = .1 )
    #print (confidence)
    output =  pred.predicted_mean
    ci = pred.conf_int()
    ci.columns = ['UpperPrice', 'LowerPrice']

    output = pd.DataFrame(output)
    output.columns = ['PredictedPrice']
    output.index.name = 'LastTradeDateKey'

    output = pd.DataFrame(output)
    output.columns = ['PredictedPrice']
    output.index.name = 'LastTradeDateKey'
    output = output.join(ci)
    output.to_csv(output_dir + symbol + '_sarimax_predictions.csv')

except Exception as e:
    logf.write("For symbol {0}, error caught. Error message: {1}\n".format(symbol, e))
