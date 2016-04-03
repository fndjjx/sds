
import os


def load_data(filename,col):
    with open(filename,'r') as f:
        lines = f.readlines()
        datalist = []
        for eachline in lines:
            eachline.strip("\n")
            datalist.append(float(eachline.split("\t")[col]))
    return datalist

def load_date_data(filename,col):
    with open(filename,'r') as f:
        lines = f.readlines()
        datalist = []
        for eachline in lines:
            eachline.strip("\n")
            datalist.append((eachline.split("\t")[col]))
    return datalist

