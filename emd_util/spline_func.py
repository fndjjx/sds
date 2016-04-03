import numpy as np
from scipy import interpolate
import copy


def spline_sample(sample_index, sample_value, raw_data, smooth_factor, order=3):

    len_raw_data = len(raw_data)
    sx = sample_index
    sy = sample_value


    rx = np.linspace(0,len_raw_data-1,len_raw_data)
    ry = interpolate.UnivariateSpline(sx,sy,s=smooth_factor,k=order)(rx)

    return ry



def test_spline_sample():
    a = [1,2,3,4,5,6,5,4,3,2,1]
    aa = [1,3,5,5,3,1]
    aaindex = [0,2,4,6,8,10]
    s = 0.1
    print spline_sample(aaindex,aa,a, s)


if __name__ == "__main__":

    test_spline_sample()


   
    

