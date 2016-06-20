
import sys

if __name__=="__main__":
    filelist = sys.argv[1].split(" ")
    
    for i in filelist:
        print i
        with open(i,'r') as f:
            lines = f.readlines()
            lastline = lines[-1]
        lastline = lastline.split(" ")
        lastline[1] = "don'tmove"
        lastline[2] = "0"
        lines.pop()
        lastline = " ".join(lastline)
        lines.append(lastline)
        with open(i,'w') as f:
            f.writelines(lines) 
