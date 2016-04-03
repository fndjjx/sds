import smtplib
from email.mime.text import MIMEText
import sys

#....
mail_to="thunderocean@163.com"
#mail_to="yans517@hotmail.com"

def send_mail(to_list,sub,content):
    #...................
    mail_host="smtp.163.com"
    mail_user="thunderocean"
    mail_pass="RHENUSmail163"
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
content = '''
def send_mail(to_list,sub,content):
    #...................
    mail_host="smtp.163.com"
    mail_user="thunderocean"
    mail_pass="RHENUSmail163"
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
'''
if __name__ == '__main__':
    mail_to="skythundergogo@163.com"
    if send_mail(mail_to,"2016-03-16 decision",content):#sys.argv[1]):
        print "nice"
    else:
        print "damn"
