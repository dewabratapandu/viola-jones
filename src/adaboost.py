import numpy as np
import math
import pandas as pd
import time
from tqdm import tqdm

def adaboost(P, N, features, hypothesis, prevCond=None, remove_indices=[]):
    """ Argumen --> P = integer jumlah image positif , N = integer jumlah image negatif
                    features = list 3D berisikan hasil hitung haar-features. Dimensi pertama terkait gambar ke-. Dimensi kedua terkait fitur ke-. Dimensi ketiga terkiat lebih detail tentang fitur yang dipakai (hitungan, type, x, y, w, h)
        Return --> strClf = list yang berisikan sejumlah weak classifier (h). setiap h memiliki nilai alfa, f, p, tetha"""
    
    # jumlah fitur
    df = pd.read_csv(features[0])
    num_features = len(df.index)
    
    if prevCond is None:
        strClf = []
        # inisialiasi bobot tiap image
        # weight positif dan negatif dalam 1D array
        weight = np.hstack((np.ones(N)*1/(2*N), np.ones(P)*1/(2*P)))
        # index fitur
        feature_index = list(range(num_features))
    else:
        strClf, weight, feature_index = prevCond
        num_features -= len(strClf)
    
    # label positif dan negatif dalam 1D array
    label = np.hstack((np.zeros(N), np.ones(P)))
    # jumlah image
    num_images = P + N
    # normalisasi weight
    weight = weight / np.sum(weight)
    # init error array
    error_classification = np.ones((num_images, num_features), dtype=np.float32)

    best_th = -1; best_error = 9999
    for t, thres in enumerate(tqdm(hypothesis)):
        if t >= num_images:
            break
        hipotesis = np.load(thres)
        if len(remove_indices) > 0:
            hipotesis = np.delete(hipotesis, remove_indices, axis=1)
            np.save(thres, hipotesis)
        
        # print('w', weight.shape, 'l', label.shape, 'h', hipotesis.shape, 't', t)
        error = map(lambda i: sum(weight[:] * np.abs(label[:] - hipotesis[i,:])), feature_index)
        error_classification[t] = np.array(list(error))
        error = np.min(error_classification[t,:])
        if error < best_error:
            best_error = error
            best_th = t
            best_hipotesis = hipotesis.copy()
    
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
    feature_index.remove(min_error_index)

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