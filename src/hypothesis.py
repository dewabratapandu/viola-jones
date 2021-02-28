import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include "stdio.h"
    #include "stdlib.h"

    __global__ void calc_hypo(float **hipo, int ***features, int thres){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
        if(features[j][i][0] < features[thres][i][0]){
            hipo[i][j] = 1;
        }
        else{
            hipo[i][j] = 0;
        }
    }
""")

def h(feature_value, threshold):
    return 1 if feature_value < threshold else 0

# def hypothesis(thres, features):
#     num_images = len(features)
#     num_features = len(features[0])

#     hipotesis = np.zeros((num_features, num_images), dtype=np.float32)
#     for i in range(num_features):
#         for j in range(num_images):
#             hipotesis[i][j] = h(features[j][i][0], features[thres][i][0])
#     return hipotesis

def hypothesis(thres, features):
    num_images = len(features)
    print('num images', num_images)
    num_features = len(features[0])
    print('num features', num_features)
    hipotesis = np.zeros((num_features, num_images), dtype=np.float32)
    
    func = mod.get_function("calc_hypo")
    func(
        cuda.Out(hipotesis), cuda.In(np.array(features)), cuda.In(np.array(thres)),
        block=(1,1,1), grid=(num_features, num_images, 1)
    )
    return hipotesis