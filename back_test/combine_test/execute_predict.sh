#!/bin/bash
date=`date +%Y%m%d`
#echo $date
python /home/ly/git_repo/my_program/sds/back_test/combine_test/real_decision.py /home/ly/git_repo/my_program/sds/back_test/combine_test/industry_stocks > /home/ly/git_repo/my_program/sds/back_test/combine_test/log_$date
