import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda import gpuarray

mod = SourceModule("""
    #include "stdio.h"
    #include "stdlib.h"

    __global__ void calc_hypo(int *features, int *thres, int *hipo){
        int i = blockIdx.x + gridDim.x * blockIdx.y;
        int th = thres[0];
        
        if(features[i] < features[th]){
            hipo[i] = 1;
        }
        else{
            hipo[i] = 0;
        }
    }
""")


# def h(feature_value, threshold):
#     return 1 if feature_value < threshold else 0

# def hypothesis(thres, features):
#     num_images = len(features)
#     num_features = len(features[0])

#     hipotesis = np.zeros((num_features, num_images), dtype=np.int8)
#     for i in range(num_features):
#         for j in range(num_images):
#             hipotesis[i][j] = h(features[j][i][0], features[thres][i][0])
#     return hipotesis

def hypothesis(thres, features):
    num_images = len(features)
    num_features = len(features[0])
    print(num_images, num_features)
    hipotesis = np.zeros((num_features * num_images), dtype=np.int8)
    features_gpu = np.array(features)[:,:,0].astype(int)
    features_gpu = features_gpu.flatten()

    hipo_gpu = gpuarray.to_gpu(hipotesis)
    features_gpu = gpuarray.to_gpu(features_gpu)
    thres_gpu = gpuarray.to_gpu(np.array(thres))

    func = mod.get_function("calc_hypo")
    func(
        features_gpu, thres_gpu, hipo_gpu,
        block=(1,1,1), grid=(num_images, num_features, 1),
    )
    hipotesis.shape = (hipotesis.size//num_features, num_features)
    hipotesis = np.transpose(hipotesis)
    return hipotesis