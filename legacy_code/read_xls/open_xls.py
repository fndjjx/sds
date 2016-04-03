
import xlrd
import sys

datafile = sys.argv[1]
data = xlrd.open_workbook(datafile)
table = data.sheets()[0]
open_price = table.col_values(i)
close_price = table.col_values(i)
macd = table.col_values(i)
