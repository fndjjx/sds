import sys
import urllib2
import os

def get_sina_data(code):
    website = "http://hq.sinajs.cn/list="
    code = str(code)
    if code.startswith("60"):
        code = "sh"+code
    else:
        code = "sz"+code

    page = website+code
    req = urllib2.Request(page)
    res = urllib2.urlopen( req )
    html=res.read()
    open_price = str(html.split(",")[1])
    current_price = str(html.split(",")[3])
    high_price = str(html.split(",")[4])
    low_price = str(html.split(",")[5])
    vol = str(html.split(",")[8])
    amount = str(html.split(",")[9])
    date = str(html.split(",")[-3])

    return open_price, high_price, low_price, current_price, vol, amount, date


if __name__=="__main__":
    r = get_sina_current_data(600816)
    print r
    
