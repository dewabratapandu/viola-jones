import numpy as np
import math
import pandas as pd
import time
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

mod = SourceModule("""
    #include "stdio.h"
    #include "stdlib.h"

    __global__ void calc_error(float **error_feature, float *weight, int *label, int *feature_index, float **hipotesis){
        int i = blockIdx.x;
        int j = blockIdx.y;
        
        int f = feature_index[i];
        error_feature[f][j] = weight[j] * abs(label[j] - hipotesis[f][j]);
    }
""")

def adaboost(P, N, features, hypothesis, prevCond=None):
    """ Argumen --> P = integer jumlah image positif , N = integer jumlah image negatif
                    features = list 3D berisikan hasil hitung haar-features. Dimensi pertama terkait gambar ke-. Dimensi kedua terkait fitur ke-. Dimensi ketiga terkiat lebih detail tentang fitur yang dipakai (hitungan, type, x, y, w, h)
        Return --> strClf = list yang berisikan sejumlah weak classifier (h). setiap h memiliki nilai alfa, f, p, tetha"""
    # label positif dan negatif dalam 1D array
    label = np.hstack((np.zeros(N), np.ones(P)))
    # jumlah image
    num_images = P + N
    # jumlah fitur
    num_features = len(features.index)
    error_classification = np.zeros((num_images, num_features), dtype=np.float32)
    
    if prevCond is None:
        strClf = []
        # inisialiasi bobot tiap image
        # weight positif dan negatif dalam 1D array
        weight = np.hstack((np.ones(N)*1/(2*N), np.ones(P)*1/(2*P)))
        # index fitur
        feature_index = list(range(num_features))
    else:
        strClf, weight, feature_index = prevCond

    # normalisasi weight
    weight = weight / np.sum(weight)

    # init errors gpu array
    error = np.zeros((num_features, num_images), dtype=np.float32)

    best_th = -1; best_error = 9999
    for t, thres in enumerate(hypothesis):
        # load hipotesis
        hipotesis = np.load(thres)

        # execute gpu computing
        func = mod.get_function("calc_error")
        func(
            cuda.Out(error), cuda.In(weight), cuda.In(label), cuda.In(np.array(feature_index)), cuda.In(hipotesis),
            block=(1,1,1), grid=(num_features, num_images,1))

        error_classification[t] = np.sum(error, axis=1)
        min_error = np.min(error_classification[t,:])
        if min_error < best_error:
            best_error = min_error
            best_th = t
            best_hipotesis = hipotesis.copy()
    
    # mencari best feature (fitur dengan error terkecil)
    feature_index = list(feature_index)
    min_error_index = np.argmin(error_classification[best_th])
    best_error = error_classification[best_th, min_error_index]
    best_feature_index = feature_index[min_error_index]
    best_hipotesis = np.load(hypothesis[best_th])
    print('min error index =', best_error)
    print('jumlah feature tersisa', len(feature_index))

    # mencari alpha dari classifier
    best_feature = (features.iloc[best_feature_index]).tolist()
    feature_weight = alpha(best_error)

    # simpan classifier dengan bobotnya
    strClf.append([best_feature, feature_weight, features[best_th][min_error_index][0]])

    # update semua bobot image yg missclassified, 
    # jika classified maka bobot * 1, jika misclassified bobot * beta
    new_weight = np.array(list(map(lambda img_index: weight[img_index] * beta(best_error) if label[img_index] == best_hipotesis[best_feature_index, img_index] else weight[img_index], range(num_images))))
    weight = new_weight
    # hilangkan fitur yang telah terpilih
    feature_index.remove(best_feature_index)

    prevCond = (strClf, weight, feature_index)
    return prevCond

def calc_errors(thres, weight, label, feature_index):
    hipotesis = np.load(thres)
    #error = map(lambda i: sum(weight[:] * np.abs(label[:] - hipotesis[i,:])), feature_index)
    error = [sum(weight[:] * np.abs(label[:] - hipotesis[i,:])) for i in feature_index]
    return np.array(error)

def alpha(error):
    return math.log(1/beta(error))

def beta(error): 
    return error/(1 - error) 