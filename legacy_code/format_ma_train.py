def format_ma_train():
    fp=open("ma_train",'r')
    lines=fp.readlines()
    fp.close()
    for line in lines:
        newline=[]
        wholeline=line.strip().split(" ")
        linelength=len(wholeline)
        format_word="%s   "%wholeline[-1]
        newline.append(format_word)
        for i in range(linelength-1):
            format_word = "%s:    %s    "%(i+1,wholeline[i])
            newline.append(format_word)
        newline.append("\n")
        fp=open("ma_train_new",'a')
        fp.writelines(newline)
    fp.close() 


if __name__=="__main__":
    format_ma_train()
    

        

