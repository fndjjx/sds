# -*- coding:utf-8 -*-  
import sys
import pandas as pd
import tushare as ts
import yaml

        
def generate_according_industry(filename, *arg):
    category = ts.get_industry_classified()
    selection = category[category["c_name"].isin(arg[0])]
    reload(sys)
    sys.setdefaultencoding('utf-8')
    selection.to_csv("tmp")
    with open("tmp") as f:
        lines = f.readlines()
        code = []
        name = []
        for index in range(len(lines)):
            if index > 0:
                code.append(lines[index].split(",")[1])
                name.append(lines[index].split(",")[2])

    code = [str(i) for i in code]
    name = [str(i) for i in name]
    stock_pair = zip(code, name)
    data = {"stock":stock_pair}

    with open(filename,'w') as f:
        yaml.dump(data, f)

def generate_according_concept(filename, filter_list):
    category = ts.get_concept_classified()
    selection = category[category["c_name"].isin(filter_list)]
    reload(sys)
    sys.setdefaultencoding('utf-8')
    selection.to_csv("tmp")
    with open("tmp") as f:
        lines = f.readlines()
        code = []
        name = []
        for index in range(len(lines)):
            if index > 0:
                code.append(lines[index].split(",")[1])
                name.append(lines[index].split(",")[2])

    code = [str(i) for i in code]
    name = [str(i) for i in name]
    stock_pair = zip(code, name)
    data = {"stock":stock_pair}

    with open(filename,'w') as f:
        yaml.dump(data, f)

def generate_according_sme(filename):
    category = ts.get_sme_classified()
    reload(sys)
    sys.setdefaultencoding('utf-8')
    category.to_csv("tmp")
    with open("tmp") as f:
        lines = f.readlines()
        code = []
        name = []
        for index in range(len(lines)):
            if index > 0:
                code.append(lines[index].split(",")[1])
                name.append(lines[index].split(",")[2])

    code = [str(i) for i in code]
    name = [str(i) for i in name]
    stock_pair = zip(code, name)
    data = {"stock":stock_pair}

    with open(filename,'w') as f:
        yaml.dump(data, f)
    

def generate_according_jeff(rawfilename,output):
    with open(rawfilename) as f:
        lines = f.readlines()
    code = []
    name = []
    for line in lines:
        code.append(line.split("\n")[0][2:])
        name.append(line.split("\n")[0][2:])

    code = [str(i) for i in code]
    name = [str(i) for i in name]
    stock_pair = zip(code, name)
    data = {"stock":stock_pair}

    with open(filename,'w') as f:
        yaml.dump(data, f)
if __name__ == "__main__":
    arg = [u"金融行业"]
  #  arg = [u"医疗器械"]
  #  arg = [u"生物制药"]
  #  arg = [u"机械行业"]
  #  arg = [u"酿酒行业"]
  #  arg = [u"传媒娱乐"]
  #  arg = [u"酒店旅游"]
  #  arg = [u"电子信息"]
    filename = "jeff_stocks"

    generate_according_jeff("jeff",filename)
    #generate_according_sme(filename)

    
        
    
