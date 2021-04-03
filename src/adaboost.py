import numpy as np
import math

def adaboost(P, N, features, thres, prevCond=None):
    """ Argumen --> P = integer jumlah image positif , N = integer jumlah image negatif
                    features = list 3D berisikan hasil hitung haar-features. Dimensi pertama terkait gambar ke-. Dimensi kedua terkait fitur ke-. Dimensi ketiga terkiat lebih detail tentang fitur yang dipakai (hitungan, type, x, y, w, h)
        Return --> strClf = list yang berisikan sejumlah weak classifier (h). setiap h memiliki nilai alfa, f, p, tetha"""
      
    if prevCond is None:
        strClf = []
        # inisialiasi bobot tiap image positif dan negatif dalam 1D array
        weight = np.hstack((np.ones(N)*1/(2*N), np.ones(P)*1/(2*P)))
    else:
        strClf, weight = prevCond

    # label positif dan negatif dalam 1D array
    label = np.hstack((np.zeros(N), np.ones(P)))
    # jumlah image dan fitur
    num_images = len(features)
    num_features = len(features[0])
    # normalisasi weight
    weight = weight / np.sum(weight)


    # calculate hypothesis
    f = np.array(features)[:,:,0].astype(int)
    hipo = np.where(f < f[thres, :], 1, 0)


    # calculate error
    error_classification = np.sum(weight * np.abs(label-np.transpose(hipo)), axis=1)

    
    # mencari best feature (fitur dengan error terkecil)
    min_error_index = np.argmin(error_classification)
    best_error = error_classification[min_error_index]
    print('best error =', best_error)

    # mencari alpha dari classifier
    feature_weight = alpha(best_error)

    # update semua bobot image yg missclassified
    b = beta(best_error)
    weight = np.where(hipo[:,min_error_index]==label, weight*b, weight)

    # simpan classifier dengan bobotnya
    best_feature = list(features[0][min_error_index])
    best_feature.append(feature_weight)
    strClf.append(best_feature)

    # hapus best feature dari features
    for i in range(num_images):
        features[i].pop(min_error_index)

    prevCond = (strClf, weight)
    return prevCond, features

def alpha(error):
    return math.log(1/beta(error))

def beta(error): 
    return error/(1 - error) 