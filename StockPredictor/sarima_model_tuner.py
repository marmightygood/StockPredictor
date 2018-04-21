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
#from matplotlib import pyplot

from sklearn.metrics import mean_squared_error
 
def parser(x):
	return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')



symbol = sys.argv[1]
input_dir = sys.argv[2]
output_dir = sys.argv[3]
model_dir = sys.argv[4]

logf = open(model_dir + "sarima_model_tuner_" + str(os.getpid()) + ".log", "a")

print ("Loading symbol {0} from {1}".format( symbol, input_dir))

#Load data
data = pd.read_csv(input_dir + symbol + '_prices.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser, usecols=[1,2])
data = data.resample('M').mean().bfill()

print (data)

parameters = pd.DataFrame( columns=['Symbol','ar','i','ma', 's','mse'])

it = 0
i=0
for ar in [1, 3, 5]:
    for ma in [0,2,4,6,8]:
        for s in [0, 1, 3, 6, 12, 24]:
            try:
                it = it+ 1
                test = data.tail(6)
                trainrows = data.count() - test.count()
                train = data.head(trainrows)

                #Fit model
                model = SARIMAX(train, order=(ar,i,ma), seasonal_order=(2,1,1,s), enforce_stationarity=False, enforce_invertibility = False)
                model_fit = model.fit()

                #Create output dataset
                output = model_fit.forecast(6)

                test = pd.DataFrame(test)
                output = pd.DataFrame(output)
                output = output.join(test)
                output.columns = ['PredictedPrice', 'ActualPrice']
                
                #Add symbol for output
                output['Symbol'] = symbol

                mse = mean_squared_error(output['PredictedPrice'], output['ActualPrice'])
                parameters.loc[it] = [symbol, ar, i, ma,s, mse]
                logf.write("For symbol {0}, processed parameters {1}, {2}, {3}, {4}\n".format(symbol, ar, i, ma, s))
            except Exception as e:
                logf.write("For symbol {0}, failed to process parameters {1}, {2}, {3}, {4}. Error message: {5}\n".format(symbol, ar, i, ma, s, e))
                pass

parameters = parameters.sort_values(by=['mse'])

# print (parameters)
#best_parameters = read_csv(input_dir + 'best_parameters.csv', index_col=0)
parameters.index.name = 'ParameterId'
parameters.to_csv (model_dir + symbol + '_sarimax_best_parameters.csv')
logf.write("For symbol {0}, tested {1} parameter combinations.\n".format(symbol, it))