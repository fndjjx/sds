
import numpy as np


class naive_bayes_classify():

    def __init__(self):
        pass

    def conditional_probability_without_laplace_smooth(self, train_input, train_output):
        
        py1 = np.sum(train_output)/float(len(train_output))
        px_y1 = []

        for i in range(train_input.shape[1]):
            px_y1.append(np.sum(train_input[:,i]*train_output)/float(np.sum(train_output)))

        train_output_oppo = []
        for i in train_output:
            if i==1:
                train_output_oppo.append(0) 
            else:
                train_output_oppo.append(1) 

        train_output_oppo = np.array(train_output_oppo) 
        px_y0 = []                    
        for i in range(train_input.shape[1]):
            px_y0.append(np.sum(train_input[:,i]*train_output_oppo)/float(np.sum(train_output_oppo)))

        return np.array(px_y0),np.array(px_y1),py1


    def conditional_probability_with_laplace_smooth(self, train_input, train_output):

        py1 = np.sum(train_output)/float(len(train_output))
        px_y1 = []

        for i in range(train_input.shape[1]):
            px_y1.append((np.sum(train_input[:,i]*train_output)+1)/(float(np.sum(train_output))+2))

        train_output_oppo = []

        for i in train_output:
            if i==1:
                train_output_oppo.append(0)
            else:
                train_output_oppo.append(1)

        train_output_oppo = np.array(train_output_oppo)
        px_y0 = []
        for i in range(train_input.shape[1]):
            px_y0.append((np.sum(train_input[:,i]*train_output_oppo)+1)/(float(np.sum(train_output_oppo))+2))

        return np.array(px_y0),np.array(px_y1),py1



    def calc_probability_with_laplace(self, px_y0, px_y1, py1, test_data):
        print px_y0
        print px_y1
        px1 = px_y1*test_data
        print px1
        product1 = 1
        for i in px1:
            if i!=0:
                product1 *= i

        px0 = px_y0*test_data
        print px0
        product0 = 1
        for i in px0:
            if i!=0:
                product0 *= i

        Numerator = product1*py1

        Denominator = (product0*(1-py1)) + (product1*py1)
  

        print product1*py1
        print product0*(1-py1)
        return Numerator/Denominator

    def calc_probability_without_laplace(self, px_y0, px_y1, py1, test_data):
        print px_y0
        print px_y1

        index = []
        for i in range(len(test_data)):
            if test_data[i] == 1:
                index.append(i)
        px1 = []
        for i in index:
            px1.append(px_y1[i])        

        product1 = 1
        for i in px1:
            product1 *= i

        ######

        px0 = []
        for i in index:
            px0.append(px_y0[i])

        product0 = 1
        for i in px0:
            product0 *= i

        Numerator = product1*py1

        Denominator = (product0*(1-py1)) + (product1*py1)


        print product1*py1
        print product0*(1-py1)
        return Numerator/Denominator


    def classify_without_laplace_smooth(self, train_input, train_output, test_data):
        px_y0, px_y1, py1 = self.conditional_probability_without_laplace_smooth(train_input, train_output)

        result = self.calc_probability_without_laplace(px_y0, px_y1, py1, test_data)

        print result
        if result > 0.5:
            print "1"
        else:
            print "0"

    def classify_with_laplace_smooth(self, train_input, train_output, test_data):
        px_y0, px_y1, py1 = self.conditional_probability_with_laplace_smooth(train_input, train_output)

        result = self.calc_probability_with_laplace(px_y0, px_y1, py1, test_data)

        print result
        if result > 0.5:
            print "1"
        else:
            print "0"

        
        
class naive_bayes_classify_multi():

    def conditional_probability(self, train_input, train_output, k):

        py = []
        px_y = []
        for i in range(k): 
            count_y = 1
            train_output_binary = []
            for j in train_output:
                if j == i:
                    count_y += 1
                    train_output_binary.append(1)
                else:
                    train_output_binary.append(0)
            py.append(count_y/float(len(train_output)))

            px_yi = []
            train_output_binary = np.array(train_output_binary)
          
            for j in range(train_input.shape[1]):
                px_yi.append((np.sum(train_input[:,j]*train_output_binary)+1)/(float(np.sum(train_output_binary)+k)))
            
            px_y.append(px_yi)

        print "px_y%s"%px_y
        print "py%s"%py
        return np.array(px_y),np.array(py)

    def calc_probability(self, px_y, py, test_data, k):
                
        px = []
        for i in range(k):
            px_yk = (px_y[i]*test_data)
            product = 1
            for j in px_yk:
                if j != 0:
                    product *= j
            px.append(product*py[i])

        print px
        px = np.array(px)
        result = px/np.sum(px)

        return result

    def classify(self, train_input, train_output, test_data, k):
        px_y, py = self.conditional_probability(train_input, train_output, k)
        result = self.calc_probability(px_y, py, test_data, k)
        print result
        

class naive_bayes_classify_event():

    def conditional_probability(self, train_input, train_output, k, dic):

        py = []
        px_y = []
        for i in range(k):
            count_y = 0
            for j in train_output:
                if j == i:
                    count_y += 1
            py.append(count_y/float(len(train_output)))

            px_yi = []
            for w in dic:
                w_sum = 0
                line_sum = 0
                for r in range(train_input.shape[0]): # for every row of input
                    if train_output[r] == i:
                        for c in range(len(train_input[r])): # for every col of input
    #                        print "begin"
    #                        print dic[train_input[r][c]]
    #                        print w
    #                        print w_sum
                            if dic[train_input[r][c]] == w:
                                w_sum += 1
    #                        print w_sum
    #                        print "end"
                        line_sum += len(train_input[r])
                px_yi.append(float(w_sum)/line_sum)

            px_y.append(px_yi)

        print "px_y%s"%px_y
        print "py%s"%py
        return np.array(px_y),np.array(py)

    def calc_probability(self, px_y, py, test_data, k):

        px = []
        for i in range(k):
            px_yk = (px_y[i]*test_data)
            product = 1
            for j in px_yk:
                if j != 0:
                    product *= j
            px.append(product*py[i])

        print px
        px = np.array(px)
        result = px/np.sum(px)

        return result

    def classify(self, train_input, train_output, test_data, k, dic):
        px_y, py = self.conditional_probability(train_input, train_output, k, dic)
        result = self.calc_probability(px_y, py, test_data, k)
        print result



#################################
#  test part
#################################

def test_naive_bayes_classify(x, y, test_data, dic):
    train_input = []
    for i in x:
        tmp = []
        for j in dic:
            if j in i:
                tmp.append(1)
            else:
                tmp.append(0)
        train_input.append(np.array(tmp))

    
    train_output = np.array(y)
    train_input = np.array(train_input)
    print train_input
    print train_output

    nb = naive_bayes_classify()
    nb.classify_without_laplace_smooth(train_input, train_output, test_data)


def test_naive_bayes_classify_laplace(x, y, test_data, dic):
    train_input = []
    for i in x:
        tmp = []
        for j in dic:
            if j in i:
                tmp.append(1)
            else:
                tmp.append(0)
        train_input.append(np.array(tmp))


    train_output = np.array(y)
    train_input = np.array(train_input)
    print train_input
    print train_output

    nb = naive_bayes_classify()
    nb.classify_with_laplace_smooth(train_input, train_output, test_data)


def test_naive_bayes_classify_multi(x, y, test_data, dic, k):
    train_input = []
    for i in x:
        tmp = []
        for j in dic:
            if j in i:
                tmp.append(1)
            else:
                tmp.append(0)
        train_input.append(np.array(tmp))


    train_output = np.array(y)
    train_input = np.array(train_input)
    print train_input
    print train_output

    nb = naive_bayes_classify_multi()
    nb.classify(train_input, train_output, test_data, k)


def test_naive_bayes_classify_event(x,y,test_data,dic,k):
    train_input = []
    for i in x:
        tmp = []
        for index,value in enumerate(dic):
            for w in i:
                if w == value:
                    tmp.append(index)
        train_input.append(np.array(tmp))

#    print dic
#    print x
#    print train_input

    train_output = np.array(y)
    train_input = np.array(train_input)
#    print train_input
#    print train_output

    nb = naive_bayes_classify_event()
    nb.classify(train_input, train_output, test_data, k, dic)




if __name__ == "__main__":
#    x = [["ab"],["ab"],["de"],["ef"],["ab"]]
#    y = [1,1,0,0,1]
#    test_data = [1,1,0,0,0,0]
#    dic = ["a","b","c","d","e","f"]
    dic = ["sunny","hot","high","weak","strong","overcast","rain","mild","cool","normal"]
    x = [["sunny","hot","high","weak"],["sunny","hot","high","strong"],["overcast","hot","high","weak"],["rain","mild","high","weak"],["rain","cool","normal","weak"],["rain","cool","normal","strong"],["overcast","cool","normal","strong"],["sunny","mild","high","weak"],["sunny","cool","normal","weak"],["rain","mild","normal","weak"],["sunny","mild","normal","strong"],["overcast","mild","high","strong"],["overcast","hot","normal","weak"],["rain","mild","high","strong"]]
#    x = [["hot","high","weak"],["hot","high","strong"],["hot","high","weak"],["mild","high","weak"],["cool","normal","weak"],["cool","normal","strong"],["overcast","cool","normal"],["mild","high","weak"],["cool","normal","weak"],["rain","mild","normal"],["mild","normal","strong"],["overcast","mild","strong"],["overcast","normal","weak"],["rain","mild","strong"]]
    y = [0,0,1,1,1,0,1,0,1,1,1,1,1,0]
    test_data = [1,0,1,0,1,0,0,0,1,0]
#    test_data = [1,1,1,1,0,0,0,0,0,0]
#    test_naive_bayes_classify(x,y, np.array(test_data),dic)
#    test_naive_bayes_classify_laplace(x,y, np.array(test_data),dic)
#    test_naive_bayes_classify_multi(x,y,np.array(test_data),dic,2)
    test_naive_bayes_classify_event(x,y,np.array(test_data),dic,2)












