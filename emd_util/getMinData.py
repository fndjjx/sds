

import sys

def getMinDataWholeday(lines, date):
    data = {}
    for line in lines:
        if line.startswith("sh") or line.startswith("sz"):
            p1 = line.split(" ")[0]
            p2 = line.split(" ")[1]
            if p1.split(",")[-1] == date:
                data[p2.split(",")[0]] = p2.split(",")[4]
    return data

def getMinDatabyTime(wholedaydata,time):
    assert type(wholedaydata) is dict
    if time in wholedaydata.keys():
        return wholedaydata[time]

def getMinDatabyFile(file_name):
    data = []
    with open(file_name) as lines:
        for line in lines:
            if line.startswith("sh") or line.startswith("sz"):
                data.append(line)
    return data
    
     
                




if __name__ == "__main__":
    file_name = sys.argv[1]
    date = sys.argv[2]
    file_data = getMinDatabyFile(file_name)
    data = getMinDataWholeday(file_data,date)
    print data
    print getMinDatabyTime(data,"14:45:00")
