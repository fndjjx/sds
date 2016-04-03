import matplotlib.pyplot as plt
import numpy as np

class LinearRegressionGradientDescent():
    
    def __init__(self, sample_predictor, sample_target):
       
        add_constant_sample_predictor = []
        for i in sample_predictor:
            new=[1]
            new.extend(i)
            add_constant_sample_predictor.append(new)

        self.predictor = np.array(add_constant_sample_predictor) 
        self.target = np.array(sample_target) 
        self.theta = self.model_generate()

        self.error = []

    def batch_gradient_descent(self, alpha, criteria):
       
        error_finish_flag = 0
        while self.cost_estimate(self.predictor,self.target,self.theta) >= criteria and (error_finish_flag == 0):
            self.error.append(self.cost_estimate(self.predictor,self.target,self.theta))
            for theta_index,theta_i in enumerate(self.theta):
                grad = [self.hypo_distance(self.predictor[j],self.target[j],self.theta)*self.predictor[j][theta_index] for j in range(len(self.predictor))]                 
                theta_i = theta_i + alpha*np.sum(grad)
                self.theta[theta_index] = theta_i
            #if len(self.error)>1: print (self.error[-1]/self.error[-2])
            if len(self.error)>1 and 0.98<(self.error[-1]/self.error[-2])<1.02:
                error_finish_flag = 1
        return self.theta

            

    def stochastic_gradient_descent(self, alpha, criteria):

        error_finish_flag = 0
        while self.cost_estimate(self.predictor,self.target,self.theta) >= criteria and (error_finish_flag == 0):
            self.error.append(self.cost_estimate(self.predictor,self.target,self.theta))
            for j in range(len(self.predictor)):
                for theta_index,theta_i in enumerate(self.theta):
                    grad = self.hypo_distance(self.predictor[j],self.target[j],self.theta)*self.predictor[j][theta_index] 
                    theta_i = theta_i + alpha*np.sum(grad)
                    self.theta[theta_index] = theta_i
            if len(self.error)>1 and 0.99<(self.error[-1]/self.error[-2])<1.01:
                error_finish_flag = 1
        return self.theta

    def cost_estimate(self, x, y, theta):
         
        matx = np.mat(x).T
        maty = np.mat(y)
        theta = np.mat(theta)
        #print theta
        #print matx
        #print maty
        #print (theta*matx)
        #print (theta*matx)-maty
        #print np.array((theta*matx)-maty)**2
        #print np.sum(np.array((theta*matx)-maty)**2)
        assert theta.shape[1] == matx.shape[0], "col of theta must == row of x"
        assert (theta*matx).shape == maty.shape, "shape of theta*matx must == shape of y"
          
        J = 0.5*np.sum(np.array((theta*matx)-maty)**2)
        print "J %s"%J
        return J

    def hypo_distance(self, x, y, theta):
                       
        matx = np.mat(x).T
        maty = np.mat(y)
        theta = np.mat(theta)
    #    print theta   
    #    print matx    
    #    print maty    
        #print (theta*matx)
        #print (theta*matx)-maty
        #print np.array((theta*matx)-maty)**2
        #print np.sum(np.array((theta*matx)-maty)**2)
        assert theta.shape[1] == matx.shape[0], "col of theta %s must == row of x %s"%(theta.shape[1],matx.shape[0])
        assert (theta*matx).shape == maty.shape, "shape of theta*matx must == shape of y"
                       
        distance = np.array((maty-(theta*matx)))
        print "distane %s"%distance
        return distance[0][0]

    def model_generate(self):
        #print self.predictor
        para_dimension = self.predictor.shape[1]
        theta = [0 for i in range(para_dimension)]
        return np.array(theta,dtype=float)
            

class LocallyWeightLinearRegressionGradientDescent(LinearRegressionGradientDescent):


    def __init__(self, sample_predictor, sample_target, core_point_index, tau):
  
        LinearRegressionGradientDescent.__init__(self,sample_predictor,sample_target)
        x = np.array(sample_predictor)
        w = ([np.exp(-(np.sum((x[core_point_index]-x[i])*(x[core_point_index]-x[i])))/(2*tau**2)) for i in range(len(x))])
        self.w = w

    def cost_estimate(self, x, y, theta):

        matx = np.mat(x).T
        maty = np.mat(y)
        theta = np.mat(theta)
        #print theta
        #print matx
        #print maty
        #print (theta*matx)
        #print (theta*matx)-maty
        #print np.array((theta*matx)-maty)**2
        #print np.sum(np.array((theta*matx)-maty)**2)
        assert theta.shape[1] == matx.shape[0], "col of theta must == row of x"
        assert (theta*matx).shape == maty.shape, "shape of theta*matx must == shape of y"

        J = 0.5*np.sum(self.w*np.array((theta*matx)-maty)**2)
        print "J %s"%J
        return J

    def hypo_distance(self, x, y, theta, w):
 
        matx = np.mat(x).T
        maty = np.mat(y)
        theta = np.mat(theta)
    #    print theta   
    #    print matx    
    #    print maty    
        #print (theta*matx)
        #print (theta*matx)-maty
        #print np.array((theta*matx)-maty)**2
        #print np.sum(np.array((theta*matx)-maty)**2)
        assert theta.shape[1] == matx.shape[0], "col of theta %s must == row of x %s"%(theta.shape[1],matx.shape[0])
        assert (theta*matx).shape == maty.shape, "shape of theta*matx must == shape of y"

        distance = w*np.array((maty-(theta*matx)))
        return distance[0][0]

    def batch_gradient_descent(self, alpha, criteria):

        error_finish_flag = 0
        while self.cost_estimate(self.predictor,self.target,self.theta) >= criteria and (error_finish_flag == 0):
            self.error.append(self.cost_estimate(self.predictor,self.target,self.theta))
            for theta_index,theta_i in enumerate(self.theta):
                grad = [self.hypo_distance(self.predictor[j],self.target[j],self.theta,self.w[j])*self.predictor[j][theta_index] for j in range(len(self.predictor))]
                theta_i = theta_i + alpha*np.sum(grad)
                self.theta[theta_index] = theta_i
            if len(self.error)>1 and 0.99<(self.error[-1]/self.error[-2])<1.01:
                error_finish_flag = 1
        return self.theta



    def stochastic_gradient_descent(self, alpha, criteria):

        error_finish_flag = 0
        while self.cost_estimate(self.predictor,self.target,self.theta) >= criteria and (error_finish_flag == 0):
            self.error.append(self.cost_estimate(self.predictor,self.target,self.theta))
            for j in range(len(self.predictor)):
                for theta_index,theta_i in enumerate(self.theta):
                    grad = self.hypo_distance(self.predictor[j],self.target[j],self.theta,self.w[j])*self.predictor[j][theta_index]
                    theta_i = theta_i + alpha*np.sum(grad)
                    self.theta[theta_index] = theta_i
            if len(self.error)>1 and 0.99<(self.error[-1]/self.error[-2])<1.01:
                error_finish_flag = 1
        return self.theta



class LogisticRegression():

    def __init__(self, sample_predictor, sample_target):

        add_constant_sample_predictor = []
        for i in sample_predictor:
            new=[1]
            new.extend(i)
            add_constant_sample_predictor.append(new)

        self.predictor = np.array(add_constant_sample_predictor)
        self.target = np.array(sample_target)
        self.theta = self.model_generate()

        self.error = []


    def stochastic_gradient_descent(self, alpha, criteria):

        error_finish_flag = 0
        while self.cost_estimate(self.predictor,self.target,self.theta) >= criteria and (error_finish_flag == 0):
            self.error.append(self.cost_estimate(self.predictor,self.target,self.theta))
            for j in range(len(self.predictor)):
                for theta_index,theta_i in enumerate(self.theta):
                    grad = self.hypo_distance(self.predictor[j],self.target[j],self.theta)*self.predictor[j][theta_index]
                    theta_i = theta_i + alpha*np.sum(grad)
                    self.theta[theta_index] = theta_i
            if 0:#len(self.error)>1 and 0.99<(self.error[-1]/self.error[-2])<1.01:
                error_finish_flag = 1
        return self.theta

    def cost_estimate(self, x, y, theta):

        matx = np.mat(x).T
        maty = np.mat(y)
        theta = np.mat(theta)
        #print theta
        #print matx
        #print maty
        #print (theta*matx)
        #print (theta*matx)-maty
        #print np.array((theta*matx)-maty)**2
        #print np.sum(np.array((theta*matx)-maty)**2)
        assert theta.shape[1] == matx.shape[0], "col of theta must == row of x"
        assert (theta*matx).shape == maty.shape, "shape of theta*matx must == shape of y"

        J = 0.5*np.sum(np.array(sigmoid_function(np.array(theta*matx))-maty)**2)
        print "J %s"%J
        return J

    def hypo_distance(self, x, y, theta):

        matx = np.mat(x).T
        maty = np.mat(y)
        theta = np.mat(theta)
    #    print theta   
    #    print matx    
    #    print maty    
        #print (theta*matx)
        #print (theta*matx)-maty
        #print np.array((theta*matx)-maty)**2
        #print np.sum(np.array((theta*matx)-maty)**2)
        assert theta.shape[1] == matx.shape[0], "col of theta %s must == row of x %s"%(theta.shape[1],matx.shape[0])
        assert (theta*matx).shape == maty.shape, "shape of theta*matx must == shape of y"

        distance = np.array((maty-sigmoid_function(theta*matx)))
        print "distance %s"%distance
        return distance[0][0]

    def model_generate(self):
        #print self.predictor
        para_dimension = self.predictor.shape[1]
        theta = [0 for i in range(para_dimension)]
        return np.array(theta,dtype=float)

def sigmoid_function(z):

    g = 1/(1+np.exp(-z))
    return g
####################
#test part
#################### 

def deco_plot(func):
    def _deco(x,y):
        theta = func(x,y)
        t = np.linspace(0,10,30)
        v = theta[0]+theta[1]*t
        plt.plot(x,y,'go')
        plt.plot(t,v,'r')
        plt.show()
    return _deco

def deco_plot_logistic(func):
    def _deco(x,y):
        theta = func(x,y)
        t = np.linspace(0,10,10)
        v1 = sigmoid_function(theta[0]+theta[1]*t)
        v = []
        for i in range(len(v1)):
            if v1[i]>0.5:
                v.append(1) 
            if v1[i]<0.5:
                v.append(0) 
        plt.plot(x,y,'go')
        plt.plot(t,v,'r')
        plt.show()
    return _deco

def test_cost_estimate(x,y,theta):
    #x = [(4,2),(1,3),(3,7)]
    #y = [5,7,8]
    #theta = [1,2]
    gd = LinearRegressionGradientDescent(x,y)
    
    print gd.cost_estimate(x,y,theta)

def test_hypo_distance(x,y,theta):
    #x = [[4],[3]]
    #y = [5,3]
    #theta = [1]
    gd = LinearRegressionGradientDescent(x,y)

    print gd.hypo_distance(x,y,theta)

@deco_plot
def test_batch_gradient_descent(x,y):

    gd = LinearRegressionGradientDescent(x,y)
    theta =  gd.batch_gradient_descent(0.0005,0.0000001)
    return theta

@deco_plot
def test_stochastic_gradient_descent(x,y):

    gd = LinearRegressionGradientDescent(x,y)
    theta =  gd.stochastic_gradient_descent(0.05,0.1)
    return theta

@deco_plot
def test_LWR_batch(x,y):

    gd = LocallyWeightLinearRegressionGradientDescent(x,y,70,1)
    theta = gd.batch_gradient_descent(0.0005,0.9)
    print theta
    return theta

@deco_plot
def test_LWR_sto(x,y):

    gd = LocallyWeightLinearRegressionGradientDescent(x,y,3,1)
    theta = gd.stochastic_gradient_descent(0.05,0.9)
    print theta
    return theta

def test_LWR_hypo_distance(x,y,theta):

    gd = LocallyWeightLinearRegressionGradientDescent(x,y)
    print gd.hypo_distance(x,y,theta,1,1)

def test_LWR_cost_estimate(x,y,theta):

    gd = LocallyWeightLinearRegressionGradientDescent(x,y)
    print gd.cost_estimate(x,y,theta,1,1)

@deco_plot_logistic
def test_logistic(x,y):
    lr = LogisticRegression(x,y)
    theta = lr.stochastic_gradient_descent(0.05,0.2)
    print theta
    print 1/(1+np.exp(-(theta[0]+theta[1]*100)))
    return theta
if __name__ == "__main__":
    #t = np.linspace(0,10,100)
    #v = np.sin(t)
    t = range(15)
    v = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
    tmp = [[i] for i in t] 
    t = tmp
    test_logistic(t,v)
    #test_batch_gradient_descent(t,v)
    #test_stochastic_gradient_descent([[1],[2],[3],[4],[5],[6]],[1,2,3,3,4,5])
    #test_hypo_distance()
    #test_LWR_batch(t,v)
    #test_LWR_sto([[1],[2],[3],[4],[5],[6]],[1,2,3,4,3,1])
    #test_LWR_hypo_distance()
    #test_LWR_cost_estimate()
    
