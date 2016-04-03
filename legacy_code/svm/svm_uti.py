
import os

def svm_predict(testdata):
    print "svm test data%s"%testdata
    svm_dir = "svm/libsvm-3.18/"
    testfile = "svm/libsvm-3.18/svm_test_data"
    moudle_file = "ma_train_new.model"
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
    print "svm test lines %s"%lines

    print testfile
    print svm_dir+moudle_file
    
    os.popen("%s/svm-predict %s %s %sout"%(svm_dir,testfile,svm_dir+moudle_file,svm_dir))
    fp=open("%s/out"%svm_dir)
    lines = fp.readlines()
    fp.close()
    print lines
    result=lines[0].strip()
    print result
    return result



if __name__=="__main__":
    testdata=[1,2,4,5]
    print svm_predict(testdata)
 
