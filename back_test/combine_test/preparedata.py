from load_data import load_data_from_tushare_pre
import sys
import yaml
import os
import time
import datetime


def removeFileInFirstDir(targetDir):
    for f in os.listdir(targetDir):
        targetFile = os.path.join(targetDir,  f)
        if os.path.isfile(targetFile):
            os.remove(targetFile)

if __name__ == "__main__":

    starttime = datetime.datetime.now()

    config_file = sys.argv[1]
    with open(config_file) as f:
        config = yaml.load(f)
    
    prefix_nofuquan = "/home/ly/git_repo/my_program/sds/back_test/combine_test/tusharedata/nofuquan/"
    prefix_fuquan = "/home/ly/git_repo/my_program/sds/back_test/combine_test/tusharedata/fuquan/"
    removeFileInFirstDir(prefix_nofuquan)
    removeFileInFirstDir(prefix_fuquan)
    for stock in config["stock"]:
        stock_number = stock[0]
        stock_name = stock[1]
        load_data_from_tushare_pre(stock_number,'2015-11-01',prefix_nofuquan,fuquan=False)
        load_data_from_tushare_pre(stock_number,'2015-11-01',prefix_fuquan,fuquan=True)

    endtime = datetime.datetime.now()
    interval=(endtime - starttime).seconds
    print interval
