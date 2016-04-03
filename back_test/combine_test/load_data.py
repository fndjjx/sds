import pandas as pd
import tushare as ts
from read_web import get_sina_data
import os
import traceback

def load_data_from_file(stock_number):
    prefix = "test_data/"
    filename = prefix + stock_number
    with open(filename,'r') as f:
        lines = f.readlines()
        close_price = []
        open_price = []
        high_price = []
        low_price = []
        vol = []
        amount = []
        for eachline in lines:
            eachline.strip("\n")
            open_price.append(float(eachline.split("\t")[1]))
            high_price.append(float(eachline.split("\t")[2]))
            low_price.append(float(eachline.split("\t")[3]))
            close_price.append(float(eachline.split("\t")[4]))
            vol.append(float(eachline.split("\t")[5]))
            amount.append(float(eachline.split("\t")[6]))

    data = {"open price":open_price, "high price":high_price, "low price":low_price, "close price":close_price, "vol":vol, "amount": amount}
    dfdata = pd.DataFrame(data)
    return dfdata


def load_data_from_tushare(stock_number, start_date, end_date):

    #stock_number = stock_number[2:]
    try:
        print "stock number for tushare {}".format(stock_number)
        raw_data = ts.get_h_data(stock_number, start = start_date, end = end_date, autype=None)
        raw_data = raw_data.sort()
        open_price = raw_data["open"]
        high_price = raw_data["high"]
        low_price = raw_data["low"]
        close_price = raw_data["close"]
        vol = raw_data["volume"]
        amount = raw_data["amount"]
        
        fuquan_data = ts.get_h_data(stock_number, start = start_date, end = end_date)
        fuquan_data = fuquan_data.sort()
        fuquan_close_price = fuquan_data["close"]
        fuquan_open_price = fuquan_data["open"]
        data = {"open price":open_price, "high price":high_price, "low price":low_price, "close price":close_price, "vol":vol, "amount": amount, "fuquan close price": fuquan_close_price, "fuquan open price": fuquan_open_price}
        dfdata = pd.DataFrame(data)
        return dfdata 
    except:
        data = {"open price":[], "high price":[], "low price":[], "close price":[], "vol":[], "amount": []}
        dfdata = pd.DataFrame(data)
        return dfdata

def load_data_from_tushare_real_time(stock_number, start_date):

    try:
        print "stock number for tushare {}".format(stock_number)
        raw_data = ts.get_h_data(stock_number, start = start_date, autype=None)
        raw_data = raw_data.sort()
        open_price = list(raw_data["open"].values)
        high_price = list(raw_data["high"].values)
        low_price = list(raw_data["low"].values)
        close_price = list(raw_data["close"].values)
        vol = list(raw_data["volume"].values)
        amount = list(raw_data["amount"].values)
        f = lambda x:str(x).split(" ")[0]
        date = map(f,list(raw_data.index))

        fuquan_data = ts.get_h_data(stock_number, start = start_date)
        fuquan_data = fuquan_data.sort()
        fuquan_close_price = list(fuquan_data["close"].values)
        fuquan_open_price = list(fuquan_data["open"].values)

        o, h, l, c, v, a, d = get_sina_data(stock_number)
        open_price.append(o) 
        close_price.append(c) 
        fuquan_close_price.append(c) 
        fuquan_open_price.append(o) 
        high_price.append(h) 
        low_price.append(l)
        vol.append(v)
        amount.append(a)
        date.append(d) 

        ff = lambda x:float(x)
        open_price = map(ff,open_price)
        high_price = map(ff,high_price)
        low_price = map(ff,low_price)
        close_price = map(ff,close_price)
        fuquan_close_price = map(ff,fuquan_close_price)
        fuquan_open_price = map(ff,fuquan_open_price)
        vol = map(ff,vol)
        amount = map(ff,amount)
        

        data = {"open price":open_price, "high price":high_price, "low price":low_price, "close price":close_price, "vol":vol, "amount": amount, "date": date, "fuquan close price": fuquan_close_price, "fuquan open price": fuquan_open_price}
        dfdata = pd.DataFrame(data)
        return dfdata
    except:
        data = {"open price":[], "high price":[], "low price":[], "close price":[], "vol":[], "amount": []}
        dfdata = pd.DataFrame(data)
        return dfdata

def removeFileInFirstDir(targetDir):
    for f in os.listdir(targetDir):
        targetFile = os.path.join(targetDir,  f)
        if os.path.isfile(targetFile):
            os.remove(targetFile)

def load_data_from_tushare_pre(stock_number, start_date, prefix, fuquan = False):

    
    datafile = prefix + stock_number
    
    
    try:
        print "stock number for tushare {}".format(stock_number)
        if not fuquan:
            raw_data = ts.get_h_data(stock_number, start = start_date, autype=None)
            raw_data = raw_data.sort()
            raw_data.to_csv(datafile)
        else:
            raw_data = ts.get_h_data(stock_number, start = start_date)
            raw_data = raw_data.sort()
            raw_data.to_csv(datafile)
    except:
        data = {"open price":[], "high price":[], "low price":[], "close price":[], "vol":[], "amount": []}
        raw_data = pd.DataFrame(data)
        raw_data.to_csv(datafile)
        
    


def load_data_from_tushare_csv_sina(stock_number):

    prefix_nofuquan = "/home/ly/git_repo/my_program/sds/back_test/combine_test/tusharedata/nofuquan/"
    prefix_fuquan = "/home/ly/git_repo/my_program/sds/back_test/combine_test/tusharedata/fuquan/"
    def read_from_csv_file(stock_number, prefix):
        datafile = prefix + stock_number
        data =  pd.read_csv(datafile)
        print data
        return data
    try:
        print "stock number for tushare {}".format(stock_number)
        raw_data = read_from_csv_file(stock_number, prefix_nofuquan)
        open_price = list(raw_data["open"].values)
        high_price = list(raw_data["high"].values)
        low_price = list(raw_data["low"].values)
        close_price = list(raw_data["close"].values)
        vol = list(raw_data["volume"].values)
        amount = list(raw_data["amount"].values)
        f = lambda x:str(x).split(" ")[0]
        date = map(f,list(raw_data.index))

        #fuquan_data = ts.get_h_data(stock_number, start = start_date)
        #fuquan_data = fuquan_data.sort()
        fuquan_data = read_from_csv_file(stock_number, prefix_fuquan)
        fuquan_close_price = list(fuquan_data["close"].values)
        fuquan_open_price = list(fuquan_data["open"].values)

        o, h, l, c, v, a, d = get_sina_data(stock_number)
        open_price.append(o)
        close_price.append(c)
        fuquan_close_price.append(c)
        fuquan_open_price.append(o)
        high_price.append(h)
        low_price.append(l)
        vol.append(v)
        amount.append(a)
        date.append(d)

        ff = lambda x:float(x)
        open_price = map(ff,open_price)
        high_price = map(ff,high_price)
        low_price = map(ff,low_price)
        close_price = map(ff,close_price)
        fuquan_close_price = map(ff,fuquan_close_price)
        fuquan_open_price = map(ff,fuquan_open_price)
        vol = map(ff,vol)
        amount = map(ff,amount)


        data = {"open price":open_price, "high price":high_price, "low price":low_price, "close price":close_price, "vol":vol, "amount": amount, "date": date, "fuquan close price": fuquan_close_price, "fuquan open price": fuquan_open_price}
        dfdata = pd.DataFrame(data)
        return dfdata
    except:
        traceback.print_exc() 
        data = {"open price":[], "high price":[], "low price":[], "close price":[], "vol":[], "amount": []}
        dfdata = pd.DataFrame(data)
        return dfdata

if __name__ == "__main__":
    data = load_data_from_tushare_real_time("601601", "2015-11-01")
    data.to_csv("tmp")
