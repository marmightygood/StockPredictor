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

#Fit model
model = ARIMA(data, order=(7,0,5))
model_fit = model.fit(disp=0)

#Create output dataset
output = model_fit.forecast(180)
prediction_series = output[0]
doutput = pd.DataFrame(data=prediction_series, columns=['PredictedPrice'])
doutput['LastTradeDateKey'] = pd.date_range(start=data._index[-1], periods=len(doutput))
doutput = doutput.set_index(['LastTradeDateKey'])
doutput.to_csv(output_dir + symbol + '_arima_predictions.csv')

# plot
pyplot.plot(data, color='blue')
pyplot.plot(doutput, color='red')
#pyplot.show()

#Add symbol for output
output['Symbol'] = symbol
output = output.set_index('Symbol', append=True)

#residuals
residuals = DataFrame(model_fit.resid)
#print(residuals.describe())
residuals.to_csv(output_dir + symbol + '_residuals.csv')
#residuals.plot()
#pyplot.show()
#residuals.plot(kind='kde')
#pyplot.show()

