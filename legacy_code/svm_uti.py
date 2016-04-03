
import os

lib_root_dir = "/root/"
#lib_root_dir = "/home/ly/"
data_root_dir = "/root/program/work/"
#data_root_dir = "/home/ly/project/my_work/"
svm_dir = lib_root_dir+"libsvm-3.18/"
data_dir = data_root_dir+"mywork_1/sds/back_test/svm/"
testfile = data_dir+"svm_test_data"
trainfile = data_dir+"svm_train_data"
scalefile = data_dir+"svm_scale_train_data"
#moudle_file = os.path.abspath('.')+"/svm_scale_train_data.model"
moudle_file = os.path.abspath('.')+"/svm_train_data.model"

def svm_train(data,n):

    generate_svm_train_file(data,n)
#    svm_scale()
    c,g=para_optimize()
    print "%s/svm-train -s 3 -c %s -g %s %s "%(svm_dir,c,g,trainfile)
    os.popen("%s/svm-train -s 3 -c %s -g %s %s "%(svm_dir,c,g,trainfile))
#    os.popen("%s/svm-train -s 4 -c %s -g %s %s "%(svm_dir,c,g,scalefile))

def para_optimize():
   
    cmd = "python %s/tools/grid2.py %s "%(svm_dir,trainfile)
    print cmd
    f = os.popen(cmd)
    lines = f.readlines()
    f.close()
    #print lines 
    c = float(lines[-1].split(" ")[0])
    g = float(lines[-1].split(" ")[1])
    #print c
    #print g
    return (c,g)

def generate_svm_train_file(data,n):
    line = []
    for i in range(len(data)-n):
        line.append(" %s "%data[n+i])
        for j in range(n):
            line.append(" %s: %s "%(j+1,data[i+j]))        
        line.append("\n")
        fp = open(trainfile,'w')
        fp.writelines(line)
        fp.close()

def svm_scale():
    os.popen("%s/svm-scale  %s > %s"%(svm_dir,trainfile,scalefile))
    
def svm_predict(testdata):
    fp = open(testfile,'w')
    line=[]
    line.append("0")
    for i in range(len(testdata)):
        line.append(" %s: %s "%(i+1,testdata[i]))
    fp.writelines(line)
    fp.close()
    fp=open(testfile)
    lines=fp.readlines()
    fp.close()
#    print "svm test lines %s"%lines

 #   print testfile
  #  print svm_dir+moudle_file
    
    os.popen("%s/svm-predict %s %s %sout"%(svm_dir,testfile,moudle_file,svm_dir))
    fp=open("%s/out"%svm_dir)
    lines = fp.readlines()
    fp.close()
    print lines
    result=lines[0].strip()
    print result
    return result


def get_svm_prediction(traindata,testdata):

    os.popen("rm %s/*"%data_dir)
    svm_train(traindata)
    return svm_predict(testdata)

if __name__=="__main__":
    f=open("emd_1000")
    lines=f.readlines()
    traindata=[]
    for line in lines:
        traindata.append(float(line.split("\n")[0]))
    svm_train(traindata)
    f.close()
    f=open("emd_1001")
    lines=f.readlines()
    testdata=[]
    for line in lines:
        testdata.append(float(line.split("\n")[0]))
    print get_svm_prediction(traindata,testdata) 
 
