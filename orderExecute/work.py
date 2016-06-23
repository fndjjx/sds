import poplib
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import urllib
from sikuli import DesktopScreenRegion
from sikuli import Pattern
from sikuli import click
from sikuli import find
from sikuli import type
import pytesseract
import os
import sys
from PIL import Image
import time
from sikuli import paste
from sikuli import exists
from sikuli import doubleClick
import smtplib
from email.mime.text import MIMEText

pathprefix = 'C:\\Users\\Administrator\\Pictures\\'
s_account = ""
s_password = ""
e_account1 = ""
e_password1 = ""
e_account2 = ""
e_password2 = ""
def login_web():

    try:
        browser = webdriver.Ie()
        targetFile = pathprefix+'untitled.png'
        if os.path.exists(targetFile):
            os.remove(targetFile)
        browser.get('https://jy.yongjinbao.com.cn/winner_gj/gjzq/')



        time.sleep(10)
        handle = browser.current_window_handle
        un = browser.find_element_by_id('account_content')
        un.clear()
        un.send_keys(s_account)


        p = Pattern(pathprefix+'3.png')
        click(p)
        click(p)
        click(p)
        type(s_password)
        #if exists("C:\Users\ly\Pictures\\passworderror4.png") or exists("C:\Users\ly\Pictures\\passworderror5.png") :
        #    browser.close()
        #    return False
        vc = browser.find_element_by_id('validateCode')



        lb = browser.find_element_by_id('sendid')


        vi = browser.find_element_by_id("imgExtraCode")

        ActionChains(browser).context_click(vi).perform()
        time.sleep(1)
        p = Pattern(pathprefix+'1.png')
        click(p)
        time.sleep(5)
        p = Pattern(pathprefix+'2.png')
        find(p)
        click(p)
        time.sleep(5)
        image = Image.open(targetFile)

        vcode = pytesseract.image_to_string(image)



        time.sleep(5)
        browser.switch_to.window(handle)
        time.sleep(5)



        vc.send_keys('{}'.format(vcode))


        lb.click()
        time.sleep(2)
        if exists(pathprefix+'vcerror.png'):
            p = Pattern(pathprefix+'confirm.png')
            click(p)
            browser.close()
            return False
        else:
    	    return True
    except:
        print sys.exc_info()[0],sys.exc_info()[1]
        if exists(pathprefix+'vcerror.png'):
            p = Pattern(pathprefix+'confirm.png')
            click(p)
        browser.close()
        return False

def buy(code, index, hold):
    print "index"
    print index
    print hold
    try:
        #p = Pattern(pathprefix+'buyselection.png')
        p = Pattern(pathprefix+'jiaoyiweituo.png')
        doubleClick(p)
        time.sleep(1)
        p = Pattern(pathprefix+'mairu.png')
        doubleClick(p)
        p = Pattern(pathprefix+'daima.png')
        click(p)
        click(p)
        click(p)
        paste(code)

        p = Pattern(pathprefix+'putongweituo.png')
        click(p)
        p = Pattern(pathprefix+'shijiacangwei.png')
        click(p)
        time.sleep(3)
        if hold == 0:
            if index == 0:
                p = Pattern(pathprefix+'13.png')
                click(p)
            elif index == 1:
                p = Pattern(pathprefix+'all.png')
                click(p)
        elif hold == 1:
            if index == 0:
                p = Pattern(pathprefix+'all.png')
                click(p)

        p = Pattern(pathprefix+'xiadan.png')
        click(p)
        time.sleep(3)
        if exists(pathprefix+'zhengquandaimabunengweikong.png'):
            p = Pattern(pathprefix+'confirm.png')
            click(p)
            raise
        p = Pattern(pathprefix+'confirm.png')
        click(p)
        return True
    except:
        return False

def sell(code):
    try:
        #p = Pattern(pathprefix+'sellselection.png')
        p = Pattern(pathprefix+'jiaoyiweituo.png')
        doubleClick(p)
        time.sleep(1)
        p = Pattern(pathprefix+'maichu.png')
        doubleClick(p)
        time.sleep(2)

        p = Pattern(pathprefix+'daima.png')
        click(p)
        click(p)
        click(p)
        paste(code)

        p = Pattern(pathprefix+'putongweituo.png')
        click(p)
        p = Pattern(pathprefix+'shijiacangwei.png')
        click(p)
        time.sleep(3)
        p = Pattern(pathprefix+'all.png')
        click(p)

        p = Pattern(pathprefix+'xiadan.png')
        click(p)
        time.sleep(3)
        if exists(pathprefix+'daimanotbeempty.png'):
            p = Pattern(pathprefix+'confirm.png')
            click(p)
            raise
        p = Pattern(pathprefix+'confirm.png')
        click(p)
        return True
    except:
        print sys.exc_info()[0],sys.exc_info()[1]
        return False

def check_hold():

    p = Pattern("C:\Users\ly\Pictures\\holding.png")
    click(p)
    time.sleep(1)
    p = Pattern("C:\Users\ly\Pictures\\holding2.png")
    click(p)
    time.sleep(1)
    if exists("C:\Users\ly\Pictures\\holding3.png"):
        return 1
    else:
        return 0



def check_new_mail():
    ISOTIMEFORMAT='%Y-%m-%d'
    current_time = time.strftime( ISOTIMEFORMAT)
    print current_time

    try:
        recvsrv = poplib.POP3('pop.163.com')
    except:
        time.sleep(10)
        recvsrv = poplib.POP3('pop.163.com')
    recvsrv.user(e_account2)
    recvsrv.pass_(e_password2)
    rsp , msg, siz = recvsrv.retr(recvsrv.stat()[0])
    print msg

    content = ""
    today_flag = False
    for i in msg:
        if i.startswith('Subject'):
            if i.split(" ")[1] == current_time:
                today_flag = True
        if i.startswith('buy') or i.startswith('sell'):
            content = i
    return today_flag,content

def analyze_mail(msg):
    bl = []
    sl = []
    csl = []
    l = msg.split(";")
    print l
    for i in l:
        print i
        if i.startswith("buy"):
            bl.append(i.split(" ")[1])
        if i.startswith("sell"):
            sl.append(i.split(" ")[1])
        if i.startswith("candidatesell"):
            csl.append(i.split(" ")[1])
    return bl,sl,csl


def start():
    login_on = False
    while login_on == False:
        login_on = login_web()
    time.sleep(10)
    send_mail(e_account1,"loginpass","loginpass")
    new_mail_arrive_flag = False
    while new_mail_arrive_flag == False:
        new_mail_arrive_flag, msg = check_new_mail()
        time.sleep(5)
    bl,sl,csl = analyze_mail(msg)
    print bl
    print sl
    print csl
    if len(sl)>0:
        for index,code in enumerate(sl):
            sell_success = False
            while sell_success == False:
                sell_success = sell(code)
        time.sleep(10)
    send_mail(e_account1,"sellpass","sellpass")
    if len(bl)>0:
        hold = len(csl)-len(sl)
        for index, code in enumerate(bl):
            buy_success = False
            #hold = check_hold()
            print hold
            while buy_success == False:
                buy_success = buy(code,index,hold)
    send_mail(e_account1,"buypass","buypass")


def send_mail(to_list,sub,content):
    #...................
    mail_host="smtp.163.com"
    mail_user=e_account1.split("@")[0]
    mail_pass=e_password1
    mail_postfix="163.com"
    me=mail_user+"<"+mail_user+"@"+mail_postfix+">"
    msg = MIMEText(content)
    msg['Subject'] = sub
    msg['From'] = me
    msg['To'] = to_list
    try:
        s = smtplib.SMTP()
        s.connect(mail_host)
        s.login(mail_user,mail_pass)
        s.sendmail(me, to_list, msg.as_string())
        s.close()
        print '1'
        return True
    except Exception, e:
        print '2'
        print str(e)
        return False
start()
