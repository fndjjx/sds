
import sys

def calc_sr(b,n,p,a):
    total=b
    for i in range(n):
        total=(1+p)*total+a

    print total


if __name__=="__main__":
    b=sys.argv[1]
    n=sys.argv[2]
    p=sys.argv[3]
    a=sys.argv[4]
    calc_sr(float(b),int(n),float(p),float(a))
