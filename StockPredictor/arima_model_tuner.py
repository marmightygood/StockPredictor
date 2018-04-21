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
model_dir = sys.argv[4]

print ("Loading symbol {0} from {1}", symbol, input_dir)

#Load data
data = pd.read_csv(input_dir + symbol + '_prices.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser, usecols=[1,2])

parameters = pd.DataFrame( columns=['Symbol','ar','i','ma','mse'])

it = 0

for ar in range(1,5):
    for i in range(0,1):
        for ma in range(0,50):
            try:
                it = it+ 1
                test = data.tail(180)
                trainrows = data.count() - test.count()
                train = data.head(trainrows)

                #Fit model
                model = ARIMA(train, order=(ar,i,ma))
                model_fit = model.fit(disp=0)

                #Create output dataset
                output = model_fit.forecast(180)
                prediction_series = output[0]
                doutput = pd.DataFrame(data=prediction_series, columns=['PredictedPrice'])
                doutput['LastTradeDateKey'] = pd.date_range(start=train._index[-1], periods=len(doutput))
                doutput = doutput.set_index(['LastTradeDateKey'])
                doutput.to_csv(output_dir + symbol + '_predictions.csv')

                # plot
                pyplot.plot(data, color='blue')
                pyplot.plot(doutput, color='red')
                pyplot.plot(test, color = 'orange')

                test = test.to_frame("TestPrice")
                doutput = doutput.join(test)
                doutput = doutput.dropna()

                #print prediction vs actual and mean error
                mse = mean_squared_error(doutput['TestPrice'], doutput['PredictedPrice'])

                #Add symbol for output
                doutput['Symbol'] = symbol
                doutput = doutput.set_index('Symbol', append=True)

                parameters.loc[it] = [symbol, ar, i, ma, mse]
            except: 
                pass
parameters = parameters.sort_values(by=['mse'])

print (parameters)
#best_parameters = read_csv(input_dir + 'best_parameters.csv', index_col=0)
parameters.to_csv (model_dir + symbol + '_best_parameters.csv')



