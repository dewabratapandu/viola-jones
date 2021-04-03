import numpy as np
import math
import pandas as pd
import time
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda import gpuarray

mod = SourceModule("""
    #include "stdio.h"
    #include "stdlib.h"

    __global__ void calc_hypo(int *features, int *thres, int *num_images, int *hipo){
        int i = blockIdx.x + gridDim.x * blockIdx.y;
        int th = thres[0];
        int num = num_iamges[0];
        
        for (int n=0; n < num; n++){
            if(features[i] < features[th]){
                hipo[i] = 1;
            }
            else{
                hipo[i] = 0;
            }
        }
    }

    __global__ void calc_error(float *weight, int *label, int *feature_index, float *hipotesis, float *error){
        int i = blockIdx.x + gridDim.x * blockIdx.y;
        int f = feature_index[i];
        error_feature[f][j] = weight[j] * abs(label[j] - hipotesis[f][j]);
    }
""")

def adaboost(P, N, features, thres, prevCond=None):
    """ Argumen --> P = integer jumlah image positif , N = integer jumlah image negatif
                    features = list 3D berisikan hasil hitung haar-features. Dimensi pertama terkait gambar ke-. Dimensi kedua terkait fitur ke-. Dimensi ketiga terkiat lebih detail tentang fitur yang dipakai (hitungan, type, x, y, w, h)
        Return --> strClf = list yang berisikan sejumlah weak classifier (h). setiap h memiliki nilai alfa, f, p, tetha"""
    # label positif dan negatif dalam 1D array
    label = np.hstack((np.zeros(N), np.ones(P)))
    # jumlah image dan fitur
    num_images = len(features)
    num_features = len(features[0])

    error_classification = np.zeros((num_images, num_features), dtype=np.float32)
    
    if prevCond is None:
        strClf = []
        # inisialiasi bobot tiap image positif dan negatif dalam 1D array
        weight = np.hstack((np.ones(N)*1/(2*N), np.ones(P)*1/(2*P)))
    else:
        strClf, weight = prevCond

    # normalisasi weight
    weight = weight / np.sum(weight)


    # execute gpu computing for hypothesis calculating
    hipotesis = np.zeros((num_images * num_features), dtype=np.int8)
    features_gpu = np.array(features)[:,:,0].astype(int)
    features_gpu = features_gpu.flatten()

    hipo_gpu = gpuarray.to_gpu(hipotesis)
    features_gpu = gpuarray.to_gpu(features_gpu)
    thres_gpu = gpuarray.to_gpu(np.array(thres))
    num_images_gpu = gpuarray.to_gpu(np.array(num_images))

    func = mod.get_function("calc_hypo")
    func(
        features_gpu, thres_gpu, num_images_gpu, hipo_gpu,
        block=(1,1,1), grid=(num_features, 1, 1),
    )
    hipotesis = hipo_gpu.get()
    hipotesis.shape = (hipotesis.size//num_features, num_features)


    # execute gpu computing for error calculating
    error = np.zeros((num_features, num_images), dtype=np.float32)
    hipo_gpu = gpuarray.to_gpu(hipotesis.flatten())

    func = mod.get_function("calc_error")
    func(
        ,
        block=(1,1,1), grid=(num_features, num_images,1)
    )

    
    # mencari best feature (fitur dengan error terkecil)
    min_error_index = np.argmin(error_classification[best_th])
    best_error = error_classification[best_th, min_error_index]
    best_feature_index = feature_index[min_error_index]
    print('best error =', best_error)
    # print('jumlah feature tersisa', len(feature_index))

    # mencari alpha dari classifier
    best_threshold = pd.read_csv(features[best_th])
    best_feature = (best_threshold.iloc[min_error_index]).tolist()
    feature_weight = alpha(best_error)

    # simpan classifier dengan bobotnya
    best_feature.extend([feature_weight, best_feature[0]])
    strClf.append(best_feature)

    # update semua bobot image yg missclassified, 
    # jika classified maka bobot * 1, jika misclassified bobot * beta
    new_weight = np.array(list(map(lambda img_index: weight[img_index] * beta(best_error) if label[img_index] == best_hipotesis[min_error_index, img_index] else weight[img_index], range(num_images))))
    weight = new_weight

    # hilangkan fitur yang telah terpilih
    # for i in range(len(features)):
    #     features[i].pop(j)

    prevCond = (strClf, weight)
    return prevCond, features

def calc_errors(thres, weight, label, feature_index):
    hipotesis = np.load(thres)
    #error = map(lambda i: sum(weight[:] * np.abs(label[:] - hipotesis[i,:])), feature_index)
    error = [sum(weight[:] * np.abs(label[:] - hipotesis[i,:])) for i in feature_index]
    return np.array(error)

def alpha(error):
    return math.log(1/beta(error))

def beta(error): 
    return error/(1 - error) 