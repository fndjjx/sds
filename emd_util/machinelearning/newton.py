
import matplotlib.pyplot as plt
import numpy as np

class Newton_simple():

    def __init__(self):

        self.root = []

    def solve(self, start_x, y, y_derivative):
        
        self.root.append(float(start_x))
        stop_criteria = 0
        while stop_criteria!=1:
            x_tmp = self.root[-1]

#            y_tmp = self.y(x_tmp)            
#            y_derivative_tmp = self.y_derivative(x_tmp)           

            y_tmp = y(x_tmp)
            y_derivative_tmp = y_derivative(x_tmp)           
        #    print "new"
        #    print x_tmp
        #    print y_tmp
        #    print y_derivative_tmp
            x_tmp = self.update(x_tmp, y_tmp, y_derivative_tmp)
            self.root.append(x_tmp)
            if self.root[-2]!=0 and 0.99<(self.root[-1]/self.root[-2])<1.01:
                stop_criteria = 1

        return self.root[-1]

    def update(self, x, y, y_derivative):

        return (x - (y/y_derivative))


#    def y(self, x):
#        return x**2+2*x-1
#
#
#    def y_derivative(self, x):
#        return 2*x+2



class LinearRegressionNewton():

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



    def newton(self, alpha, criteria):

        error_finish_flag = 0
        while self.cost_estimate(self.predictor,self.target,self.theta) >= criteria and (error_finish_flag == 0):
            self.error.append(self.cost_estimate(self.predictor,self.target,self.theta))
            for j in range(len(self.predictor)):
                for theta_index,theta_i in enumerate(self.theta):
                    grad = self.hypo_distance(self.predictor[j],self.target[j],self.theta)
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

        H = np.mat([[x[0], x[1]],[x[1]*x[0],x[1]*x[1]]]) #Get l(theta) Hessian matrix
        matx = np.mat(x).T
        maty = np.mat(y)
        delta = np.mat(np.array((maty-(theta*matx)))*x)

        t = H.I*delta.T
        return t

    def model_generate(self):
        #print self.predictor
        para_dimension = self.predictor.shape[1]
        theta = [0 for i in range(para_dimension)]
        return np.array(theta,dtype=float)










######
# test part
#####


def test_Newton_simple(x,y,y_derivative):
    n = Newton_simple()
    answer = n.solve(x,y,y_derivative)

    print answer


def deco_plot(func):
    def _deco(x,y):
        theta = func(x,y)
        t = np.linspace(0,10,30)
        v = theta[0]+theta[1]*t
        plt.plot(x,y,'go')
        plt.plot(t,v,'r')
        plt.show()
    return _deco


@deco_plot
def test_linear_regression_newton(x,y):

    gd = LinearRegressionNewton(x,y)
    theta =  gd.newton(0.05,0.1)
    return theta


if __name__ == "__main__":

#    y = lambda x:x*100+x**3+x**8-2+np.exp(x)
#    y_derivative = lambda x:100+3*x**2+8*x**7+np.exp(x)
#    test_Newton_simple(-5,y,y_derivative)

    
    test_linear_regression_newton([[2.1],[3.1],[4.1],[5.1],[6.9],[7.0]],[2,3,3,4,5,2]) 
        
