import pypyodbc
import pandas.io.sql as psql
import pandas as pd

conn = pypyodbc.connect("DRIVER={SQL Server};SERVER=localhost;UID=analysis_services;PWD=DarthV;DATABASE=Stocks")
frame = pd.read_sql('SELECT TOP 100 * FROM Star.FactPriceSnapshot', conn)

print (frame)