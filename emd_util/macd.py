
import sys
sys.setrecursionlimit(4000000)

def ema(data,index,n):
    if index == n:
        return ((n-1)*(float(sum(data[:n]))/n)/(n+1))+(2*data[index]/(n+1))
        #return float(sum(data[:n]))/n
    else:
        return (((n-1)*ema(data,index-1,n)+2*data[index])/(n+1))




    



if __name__ == "__main__":

    datafile = sys.argv[1]
    fp = open(datafile)
    lines = fp.readlines()
    fp.close()
    close_price = []
    high_price = []
    low_price = []
    open_price = []
    date = []
    macd = []
    for eachline in lines:
        eachline.strip()
        date.append(eachline.split("\t")[0])
        open_price.append(float(eachline.split("\t")[1]))
        close_price.append(float(eachline.split("\t")[4]))
        high_price.append(float(eachline.split("\t")[2]))
        low_price.append(float(eachline.split("\t")[3]))
    print open_price[-1]
    print high_price[-1]
    print low_price[-1]
    print close_price[-1]
    di = [(high_price[i]+low_price[i]+2*close_price[i])/4 for i in range(len(close_price))]
    ema12 = [ema(di,i,12) for i in range(12,len(di))][14:]
    ema26 = [ema(di,i,26) for i in range(26,len(di))]
    print ema12[-1]-ema26[-1]
    dif = [ema12[i]-ema26[i] for i in range(len(ema26))]
    dea9 = [ema(dif,i,9) for i in range(9,len(dif))]

    di_now = (high_price[-1]+low_price[-1]+2*close_price[-1])/4
    ema12_now  = (ema12[-2]*11)/13+(di_now*2)/13
    ema26_now  = (ema26[-2]*25)/27+(di_now*2)/27
    dif_now = ema12_now-ema26_now
    dea_now = (dea9[-2]*8)/10+(dif_now*2)/10
    macd = dif_now - dea_now
    print macd


    newline = str(date[-1])+"\t"+str(open_price[-1])+"\t"+str(high_price[-1])+"\t"+str(low_price[-1])+"\t"+str(close_price[-1])+"\t"+str(macd)
    fp = open(datafile,'r')
    lines = fp.readlines()
    fp.close()
    lines.pop()
    lines.append(newline)
    fp = open(datafile,'w')
    fp.writelines(lines)
    fp.writelines("\n")
    fp.close()
