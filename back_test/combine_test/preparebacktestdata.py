import MySQLdb as md
import tushare as ts
import pandas as pd
import yaml
import sys

start_date = '2000-10-05'
end_date = '2016-06-18'

def get_tushare_data(code, start_date, end_date, fuquan=False):
    print "code {}".format(code)
    if fuquan:
        return ts.get_h_data(code, start = start_date, end = end_date)
    else:
        return ts.get_h_data(code, start = start_date, end = end_date, autype=None)


def open_db(user,passwd,dbname):
    cxn = md.connect(db=dbname, user=user, passwd=passwd)
    return cxn



def save_to_sql(conn, tablename, df):
    df=df.sort()
    index = pd.DataFrame(df.index.values,index=df.index,columns=["date"] )
    df = df.join(index)
    df.to_sql(con=conn, name=tablename, if_exists='replace', flavor='mysql')



def read_sql(conn,tablename):
    sql = 'select * from {}'.format(tablename)
    return pd.read_sql(sql,conn)

def save(code,start_date,end_date,fuquan=False):
    if fuquan:
        tablename = code+'fuquan'
        conn = open_db('root','root','ts_db')
        save_to_sql(conn,tablename,get_tushare_data(code,start_date,end_date,fuquan))
    else:
        tablename = code+'nofuquan'
        conn = open_db('root','root','ts_db')
        save_to_sql(conn,tablename,get_tushare_data(code,start_date,end_date,fuquan))



def download_data_to_sql(config_file):
    with open(config_file) as f:
        config = yaml.load(f) 
    for stock in config["stock"]:
        stock_number = stock[0]
        stock_name = stock[1]
        save(stock_number,start_date,end_date,fuquan=False)
        save(stock_number,start_date,end_date,fuquan=True)


if __name__ == "__main__":
    config_file = sys.argv[1]
    download_data_to_sql(config_file)
    
